"""One bounded diagnostic batch inside the existing $5 reservation."""
import os
os.environ['MODAL_PROFILE']='j999ucl'
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import modal
sys.path.insert(0,'/opt/scene')
from cloud import image,volume

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'.runtime/rack-investigation' if modal.is_local() else Path('/opt/experiment-inputs')
app=modal.App('agverase-rack-investigation')
experiment_image=image.add_local_dir(str(INPUT),'/opt/experiment-inputs',ignore=['cloud','*.log'])

@app.function(image=experiment_image,gpu='A10',cpu=(2,4),memory=(32768,65536),timeout=240,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2,volumes={'/data':volume})
def experiment(run_id):
    root=Path('/data/investigations')/run_id;root.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+215
    env=dict(os.environ,PYTHONPATH='/opt/openpi/third_party/libero:/opt/openpi/packages/openpi-client/src:/opt/scene')
    try:
        with (root/'wrapper.log').open('w') as log:
            result=subprocess.run(['/opt/libero-env/bin/python','/opt/scene/investigate_wrapper.py'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        print((root/'wrapper.log').read_text()[-1600:],flush=True)
        if result.returncode:raise RuntimeError('Wrapper comparison failed')
        import shutil
        shutil.copy2('/tmp/wrapper-result.json',root/'wrapper-result.json')
        for name in ['investigate_cloud.py','investigate_policy.py','investigate_wrapper.py','investigate_rack.py']:
            shutil.copy2(Path('/opt/scene')/name,root/name)
        with (root/'policy.log').open('w') as log:
            result=subprocess.run(['/opt/openpi/.venv/bin/python','-u','/opt/scene/investigate_policy.py','/opt/experiment-inputs',str(root)],cwd='/opt/openpi',env=dict(os.environ,PYTHONPATH='/opt/openpi/src:/opt/openpi/packages/openpi-client/src'),stdout=log,stderr=subprocess.STDOUT,timeout=max(10,int(deadline-time.monotonic())))
        print((root/'policy.log').read_text()[-3000:],flush=True)
        if result.returncode:raise RuntimeError('Policy input experiment failed')
    except Exception as exc:
        (root/'error.json').write_text(json.dumps({'error':repr(exc)}));raise
    finally:volume.commit()
    return str(root)


def main():
    ledger=HERE.parent/'.runtime/mujoco-agent/budget.json';budget=json.loads(ledger.read_text());entry=budget['runs'][-1]
    if entry.get('investigation_attempts',0)>=2:raise RuntimeError('Diagnostic allowance exhausted')
    # Verified last container lifetime: 03:48:37–03:54:36 (359 seconds).
    # Keep 600 seconds; release only 480 of its original 1080-second reservation.
    if not any(r.get('attempt')==8 for r in entry['additional_reconciliations']):
        entry['additional_reconciliations'].append({'attempt':8,'reserved_seconds':1080,'conservative_used_seconds':600,'released_seconds':480,'gpu_started_utc':'2026-11-08 03:48:37','app_stopped_utc':'2026-11-08 03:54:36','reason':'Completed native trials; unused reservation funds user-requested input experiments'})
        entry['gpu_seconds_reserved']-=480
    if entry.get('investigation_attempts')==1:
        if entry.get('investigation_status')!='failed':raise RuntimeError('Previous diagnostic job is not stopped')
        entry['investigation_startup_failure']={'app_id':'ap-zWSWv0ExCXnzl3r3EKHLgr','created_utc':'2026-11-08 04:15:46','stopped_utc':'2026-11-08 04:16:29','retained_seconds':120,'released_unexecuted_function_seconds':360}
        entry['gpu_seconds_reserved']-=360
    if entry['gpu_seconds_reserved']+360>7200:raise RuntimeError('Existing compute allowance is insufficient')
    assert budget['reserved']==5 and entry['reserved_usd']==5
    entry['gpu_seconds_reserved']+=360;entry['investigation_attempts']=entry.get('investigation_attempts',0)+1;entry['investigation_status']='starting'
    ledger.write_text(json.dumps(budget,indent=2));run_id=time.strftime('%Y%m%d-%H%M%S')
    try:
        with modal.enable_output(),app.run():
            entry['investigation_app_id']=app.app_id;ledger.write_text(json.dumps(budget,indent=2))
            watchdog=threading.Timer(1080,lambda:subprocess.run([sys.executable,'-m','modal','app','stop',app.app_id,'--yes']))
            watchdog.daemon=True;watchdog.start()
            try:
                call=experiment.spawn(run_id);entry['investigation_call_id']=call.object_id;ledger.write_text(json.dumps(budget,indent=2))
                since=time.monotonic()
                while experiment.get_current_stats().num_running_inputs<1:
                    try:call.get(timeout=0);break
                    except TimeoutError:pass
                    if time.monotonic()-since>600:call.cancel(terminate_containers=True);raise TimeoutError('GPU scheduling exceeded ten minutes')
                    time.sleep(2)
                remote=call.get()
            finally:watchdog.cancel()
        entry['investigation_status']='completed'
    except BaseException:
        entry['investigation_status']='failed';raise
    finally:ledger.write_text(json.dumps(budget,indent=2))
    (INPUT/'cloud-result.json').write_text(json.dumps({'remote':remote,'run_id':run_id}))
    destination=INPUT/'cloud';destination.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','modal','volume','get','--force','agverase-mujoco-episodes',remote.removeprefix('/data/')+'/',str(destination)],check=True)

if __name__=='__main__':main()
