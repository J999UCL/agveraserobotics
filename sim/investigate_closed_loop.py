"""Controlled scene/language experiments using the existing native rollout API."""
import os
os.environ['MODAL_PROFILE']='j999ucl'
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
import modal
sys.path.insert(0,'/opt/scene')
from cloud import image,volume

HERE=Path(__file__).resolve().parent
app=modal.App('agverase-rack-closed-loop')

@app.function(image=image,gpu='A10',cpu=(2,4),memory=(32768,65536),timeout=480,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,volumes={'/data':volume})
def experiment(run_id,mode="clutter",max_seconds=480):
    root=Path('/data/investigations')/run_id;root.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+max_seconds-30
    assets=Path('/opt/scene/tasks/assets');variant=root/('forward-target-rack' if mode=='pose' else 'single-visible-rack');variant.mkdir()
    for path in assets.glob('rack_*.xml'):
        xml=ET.parse(str(path))
        for element in xml.findall('.//*[@file]'):element.set('file',str((assets/element.get('file')).resolve()))
        if mode=='pose' and path.name=='rack_b.xml':
            body=xml.find('.//body[@name="object"]');position=list(map(float,body.get('pos').split()));position[1]+=.18;body.set('pos',' '.join(map(str,position)))
        if mode=='clutter' and path.name!='rack_b.xml':
            for geom in xml.findall('.//geom'):
                if geom.get('group')=='1':geom.set('rgba','0 0 0 0')
        xml.write(str(variant/path.name),encoding='unicode')
    import shutil
    for name in ['libero_client.py','libero_task.py','investigate_closed_loop.py']:
        shutil.copy2(Path('/opt/scene')/name,root/name)
    native_env=dict(os.environ,PYTHONPATH='/opt/openpi/third_party/libero:/opt/openpi/packages/openpi-client/src:/opt/scene')
    policy_log=(root/'policy.log').open('w')
    server=subprocess.Popen(['/opt/openpi/.venv/bin/python','/opt/openpi/scripts/serve_policy.py','--env','LIBERO','--port','8000','--record'],cwd=str(root),env=dict(os.environ,PYTHONPATH='/opt/openpi/src:/opt/openpi/packages/openpi-client/src'),stdout=policy_log,stderr=subprocess.STDOUT)
    summary={}
    try:
        with (root/'ready.log').open('w') as log:
            subprocess.run(['/opt/libero-env/bin/python','-c',"from openpi_client.websocket_client_policy import WebsocketClientPolicy; WebsocketClientPolicy('127.0.0.1',8000); print('ready')"],env=native_env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=100)
        cases=[('original','Pull out the server with the cyan handle.',None),('wording','open the middle drawer of the cabinet',None),('clutter_wording','open the middle drawer of the cabinet',str(variant))]
        if mode=='pose':cases=[('pose_server','Pull out the server with the cyan handle.',str(variant)),('pose_drawer','open the middle drawer of the cabinet',str(variant))]
        for name,prompt,asset_path in cases:
            env=dict(native_env)
            if asset_path:env['AGVERASE_RACK_ASSETS']=asset_path
            output=root/name
            with (root/(name+'.log')).open('w') as log:
                result=subprocess.run(['/opt/libero-env/bin/python','-u','/opt/scene/libero_client.py','run',str(output),'--episodes','2','--max-steps','300','--prompt',prompt],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(10,int(deadline-time.monotonic())))
            print((root/(name+'.log')).read_text()[-2000:],flush=True)
            if result.returncode:raise RuntimeError(name+' rollout failed')
            summary[name]=json.loads((output/'results.json').read_text())
            (root/'summary.json').write_text(json.dumps(summary,indent=2));volume.commit()
        (root/'status.json').write_text(json.dumps({'status':'completed','method':'native closed loop; 2 trials x 300 policy actions per case; seeds=7','mode':mode,'target_rack_y_offset_m':.18 if mode=='pose' else 0}))
    except Exception as exc:
        (root/'error.json').write_text(json.dumps({'error':repr(exc)}));raise
    finally:
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        policy_log.close();volume.commit()
    return str(root)


def main():
    runtime=HERE.parent/'.runtime/rack-investigation';ledger=HERE.parent/'.runtime/mujoco-agent/budget.json'
    budget=json.loads(ledger.read_text());entry=budget['runs'][-1]
    if entry.get('closed_loop_experiment'):raise RuntimeError('Closed-loop diagnostic allowance already used')
    assert budget['reserved']==5 and entry['investigation_status']=='completed'
    verified=json.loads((runtime/'verified-app-times.json').read_text())
    updates=[(5,'2026-11-08 02:44:22','2026-11-08 02:49:20',360),(6,'2026-11-08 02:50:16','2026-11-08 02:57:10',480),(7,'2026-11-08 03:41:23','2026-11-08 03:45:51',360),(8,'2026-11-08 03:48:37','2026-11-08 03:54:36',420)]
    reconciliations=[]
    for attempt,start,stop,retained in updates:
        matched=[a for a in verified if a['description']=='agverase-mujoco-agent' and a['state']=='stopped' and a['stopped_at'].startswith(stop.replace(' ','T'))]
        # CLI uses a space separator rather than ISO T.
        if not matched:matched=[a for a in verified if a['description']=='agverase-mujoco-agent' and a['state']=='stopped' and a['stopped_at'].startswith(stop)]
        if not matched:raise RuntimeError('Missing verified stop time for attempt '+str(attempt))
        from datetime import datetime
        elapsed=(datetime.fromisoformat(stop)-datetime.fromisoformat(start)).total_seconds()
        assert retained>=elapsed+60
        reconciliations.append({'attempt':attempt,'previous_retained_seconds':600,'new_retained_seconds':retained,'released_seconds':600-retained,'gpu_container_started_utc':start,'app_stopped_utc':stop,'verified_app_id':matched[0]['app_id'],'reason':'Refine completed-job upper bound with at least sixty seconds of additional margin'})
    if entry.get('refined_timing_reconciliations'):raise RuntimeError('Timing reconciliation already applied')
    new_total=entry['gpu_seconds_reserved']-sum(r['released_seconds'] for r in reconciliations)+600
    if new_total>7200:raise RuntimeError('Cumulative resource allowance exhausted')
    entry['refined_timing_reconciliations']=reconciliations;entry['gpu_seconds_reserved']=new_total
    entry['closed_loop_experiment']={'status':'starting','reserved_seconds':600}
    ledger.write_text(json.dumps(budget,indent=2));run_id=time.strftime('%Y%m%d-%H%M%S')+'-closed-loop'
    try:
        with modal.enable_output(),app.run():
            entry['closed_loop_experiment']['app_id']=app.app_id;ledger.write_text(json.dumps(budget,indent=2))
            watchdog=threading.Timer(1200,lambda:subprocess.run([sys.executable,'-m','modal','app','stop',app.app_id,'--yes']));watchdog.daemon=True;watchdog.start()
            try:
                call=experiment.spawn(run_id);since=time.monotonic()
                while experiment.get_current_stats().num_running_inputs<1:
                    try:call.get(timeout=0);break
                    except TimeoutError:pass
                    if time.monotonic()-since>600:call.cancel(terminate_containers=True);raise TimeoutError('GPU scheduling exceeded ten minutes')
                    time.sleep(2)
                remote=call.get()
            finally:watchdog.cancel()
        entry['closed_loop_experiment']['status']='completed';entry['closed_loop_experiment']['remote']=remote
    except BaseException:
        entry['closed_loop_experiment']['status']='failed';raise
    finally:ledger.write_text(json.dumps(budget,indent=2))
    (runtime/'closed-loop-result.json').write_text(json.dumps({'remote':remote,'run_id':run_id}))
    # Retrieve diagnostics first; standard episode recordings remain available on the volume.
    destination=runtime/'closed-loop';destination.mkdir(exist_ok=True)
    for name in ['summary.json','status.json','original.log','wording.log','clutter_wording.log']:
        subprocess.run([sys.executable,'-m','modal','volume','get','--force','agverase-mujoco-episodes',remote.removeprefix('/data/')+'/'+name,str(destination/name)],check=True)

def main_pose():
    runtime=HERE.parent/'.runtime/rack-investigation';ledger=HERE.parent/'.runtime/mujoco-agent/budget.json'
    budget=json.loads(ledger.read_text());entry=budget['runs'][-1]
    if entry.get('pose_experiment'):raise RuntimeError('Pose diagnostic allowance already used')
    assert budget['reserved']==5 and entry['closed_loop_experiment']['status']=='completed'
    verified=json.loads((runtime/'final-apps.json').read_text())
    assert any(a['app_id']==entry['closed_loop_experiment']['app_id'] and a['state']=='stopped' and a['stopped_at'].startswith('2026-11-08 04:28:59') for a in verified)
    # Verified lifetime 04:24:34–04:28:59 = 265 s; retain 360 s, release 240.
    total=entry['gpu_seconds_reserved']-240+420
    if total>7200:raise RuntimeError('Existing resource allowance exhausted')
    entry['closed_loop_timing_reconciliation']={'previous_reserved_seconds':600,'conservative_used_seconds':360,'released_seconds':240,'gpu_started_utc':'2026-11-08 04:24:34','app_stopped_utc':'2026-11-08 04:28:59'}
    entry['gpu_seconds_reserved']=total;job={'status':'starting','reserved_seconds':420};entry['pose_experiment']=job
    ledger.write_text(json.dumps(budget,indent=2));run_id=time.strftime('%Y%m%d-%H%M%S')+'-pose'
    try:
        with modal.enable_output(),app.run():
            job['app_id']=app.app_id;ledger.write_text(json.dumps(budget,indent=2))
            watchdog=threading.Timer(1080,lambda:subprocess.run([sys.executable,'-m','modal','app','stop',app.app_id,'--yes']));watchdog.daemon=True;watchdog.start()
            try:
                worker=experiment.with_options(timeout=300);call=worker.spawn(run_id,'pose',300);since=time.monotonic()
                while worker.get_current_stats().num_running_inputs<1:
                    try:call.get(timeout=0);break
                    except TimeoutError:pass
                    if time.monotonic()-since>600:call.cancel(terminate_containers=True);raise TimeoutError('GPU scheduling exceeded ten minutes')
                    time.sleep(2)
                remote=call.get()
            finally:watchdog.cancel()
        job.update(status='completed',remote=remote)
    except BaseException:
        job['status']='failed';raise
    finally:ledger.write_text(json.dumps(budget,indent=2))
    (runtime/'pose-result.json').write_text(json.dumps({'remote':remote,'run_id':run_id}))
    destination=runtime/'pose';destination.mkdir(exist_ok=True)
    for name in ['summary.json','status.json','pose_server.log','pose_drawer.log']:
        subprocess.run([sys.executable,'-m','modal','volume','get','--force','agverase-mujoco-episodes',remote.removeprefix('/data/')+'/'+name,str(destination/name)],check=True)

if __name__=='__main__':
    if '--pose' in sys.argv:main_pose()
    else:main()
