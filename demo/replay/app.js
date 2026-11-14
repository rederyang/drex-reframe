const $=id=>document.getElementById(id);
let manifest,record,frame=0,stage='crop',inspector='request',imageSequence=0,loadSequence=0;
const timecode=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(1).padStart(4,'0')}`;
const safeOption=(value,label)=>{const o=document.createElement('option');o.value=value;o.textContent=label;return o;};
const videoSource=$('source-video'),videoPortrait=$('portrait-video');
function showError(error){$('error').hidden=false;$('error').textContent=`Could not load this recorded run. ${error.message||''}`;}
async function getJSON(path){const r=await fetch(path);if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json();}
function setStage(value){stage=value;render();}
function clip(){return manifest.clips.find(c=>c.id===$('clip').value);}
async function chooseClip(){
  const cases=manifest.cases.filter(c=>c.clip_id===$('clip').value);
  $('focus').replaceChildren(...cases.map(c=>safeOption(c.id,c.focus)));
  await chooseCase();
}
async function chooseCase(){
  const n=++loadSequence;
  const entry=manifest.cases.find(c=>c.id===$('focus').value);
  const loaded=await getJSON(entry.record);if(n!==loadSequence)return;
  record=loaded;frame=0;stage='crop';$('error').hidden=true;
  videoSource.pause();videoPortrait.pause();
  videoSource.src=clip().video;videoPortrait.src=record.video;
  $('download-video').href=record.video;
  $('download-json').href=entry.record;$('download-json').download=`${record.id}.json`;
  $('goal').textContent=record.goal||'Keep the visible play in context.';
  const seconds=record.response._latency_s;
  $('metrics').textContent=`8 frames · ${Object.keys(record.request.questions).length} choices · ${Number(seconds).toFixed(2)}s recorded`;
  $('frames').replaceChildren(...record.frames.map((f,i)=>{const b=document.createElement('button');b.textContent=timecode(f.t);b.setAttribute('aria-label',`Frame ${timecode(f.t)}`);b.addEventListener('click',()=>{frame=i;render();});return b;}));
  render();
}
async function renderCanvas(){
  if(!record)return;
  const n=++imageSequence,img=new Image();
  img.src=stage==='request'?record.frames[frame].input:record.frames[frame].source;
  await img.decode();if(n!==imageSequence)return;
  const canvas=$('canvas'),ctx=canvas.getContext('2d');
  const row=record.keyframes[frame],W=row.kf_W,H=row.kf_H;
  canvas.width=W;canvas.height=H;ctx.drawImage(img,0,0,W,H);
  if(stage==='crop'||stage==='response'){
    const dec=record.decoded[frame],[x0,y0,x1,y1]=dec.region;
    ctx.strokeStyle='#ffe600';ctx.lineWidth=3;ctx.strokeRect(x0,y0,x1-x0,y1-y0);
    if(dec.focus){ctx.beginPath();ctx.arc(dec.focus.x,dec.focus.y,7,0,2*Math.PI);ctx.fillStyle='#ffe600';ctx.fill();ctx.strokeStyle='#18251b';ctx.lineWidth=2;ctx.stroke();}
    if(stage==='crop'){
      const box=record.frames[frame].crop_box;
      if(box){ctx.strokeStyle='#00dcff';ctx.lineWidth=4;ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);}
    }
  }
}
function renderChart(){
  if(!record)return;
  const name=$('question').value,answer=record.response.answers[`f${frame+1}_${name}`];
  const pairs=Object.entries(answer.probabilities||{}),peak=Math.max(...pairs.map(([,p])=>p));
  $('answer').textContent=`Recorded choice: ${answer.choice}`;
  $('chart').replaceChildren(...pairs.map(([label,p])=>{const wrap=document.createElement('div');wrap.className='bar-wrap';wrap.title=`${label}: ${(p*100).toFixed(2)}%`;const bar=document.createElement('div');bar.className=`bar${p===peak?' peak':''}`;bar.style.height=`${Math.max(2,p*100)}%`;const caption=document.createElement('small');caption.textContent=label;wrap.append(bar,caption);return wrap;}));
}
function renderInspector(){
  if(!record)return;
  const value=inspector==='decoded'?record.decoded[frame]:record[inspector];
  $('json').textContent=JSON.stringify(value,null,2);
  $('inspector-label').textContent=inspector==='request'?'POST /v1/systemone · actual recorded input (image bytes available as JPEG files)':inspector==='response'?'Actual NACE response':inspector==='decoded'?'main decoder · probability floor correction → expected coordinates + focus':'main camera planner · smoothing + focus constraints';
  document.querySelectorAll('[data-inspect]').forEach(b=>b.classList.toggle('active',b.dataset.inspect===inspector));
  $('provenance').textContent=`main ${record.pipeline_revision.slice(0,7)} · request ${record.request_sha256.slice(0,16)}`;
}
function render(){
  if(!record)return;
  document.querySelectorAll('[data-stage]').forEach(b=>{b.classList.toggle('active',b.dataset.stage===stage);b.setAttribute('aria-pressed',String(b.dataset.stage===stage));});
  [...$('frames').children].forEach((b,i)=>{b.classList.toggle('active',i===frame);b.setAttribute('aria-pressed',String(i===frame));});
  const video=stage==='video';
  $('canvas').hidden=video;$('portrait').hidden=video;videoSource.hidden=!video;videoPortrait.hidden=!video;
  if(!video){videoSource.pause();videoPortrait.pause();renderCanvas().catch(showError);$('portrait').src=record.frames[frame].portrait;}
  $('source-label').textContent=stage==='request'?'Recorded NACE input':'Source';
  $('legend').style.visibility=stage==='crop'||stage==='response'?'visible':'hidden';
  $('response-panel').hidden=stage!=='response';
  if(stage==='response')renderChart();
  renderInspector();
}
$('clip').addEventListener('change',()=>chooseClip().catch(showError));
$('focus').addEventListener('change',()=>chooseCase().catch(showError));
$('question').addEventListener('change',renderChart);
document.querySelectorAll('[data-stage]').forEach(b=>b.addEventListener('click',()=>setStage(b.dataset.stage)));
document.querySelectorAll('[data-inspect]').forEach(b=>b.addEventListener('click',()=>{inspector=b.dataset.inspect;renderInspector();}));
$('replay').addEventListener('click',()=>{setStage('video');videoSource.currentTime=0;videoPortrait.currentTime=0;videoSource.play().catch(()=>{});});
videoSource.addEventListener('play',()=>{videoPortrait.currentTime=videoSource.currentTime;videoPortrait.play().catch(()=>{});});
videoSource.addEventListener('pause',()=>videoPortrait.pause());
videoSource.addEventListener('seeked',()=>{videoPortrait.currentTime=videoSource.currentTime;});
try{manifest=await getJSON('manifest.json');$('clip').replaceChildren(...manifest.clips.map(c=>safeOption(c.id,c.label)));await chooseClip();}catch(error){showError(error);}
