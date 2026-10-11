
'use strict';
const data=JSON.parse(document.getElementById('event-data').textContent);
let selected=0;
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=2)=>x===null||x===undefined?'—':Number(x).toFixed(n);
const moneyClass=x=>x===null?'':x>0?'positive':x<0?'negative':'';
const local=x=>x?new Date(x).toLocaleString('sv-SE',{timeZone:'America/New_York',hour12:false}):'unknown';
function renderRows(){
  const q=document.getElementById('search').value.toLowerCase().trim();
  let rows=data.map((e,i)=>({e,i})).filter(({e})=>`${e.timestamp_et} ${e.strategy} ${e.setup_type} ${e.setup_name} ${e.setup_side>0?'long':'short'}`.toLowerCase().includes(q));
  const sort=document.getElementById('sort').value;
  if(sort!=='time')rows.sort((a,b)=>{if(a.e.net_points_12===null)return 1;if(b.e.net_points_12===null)return -1;return sort==='best'?b.e.net_points_12-a.e.net_points_12:a.e.net_points_12-b.e.net_points_12});
  document.getElementById('events').innerHTML=rows.map(({e,i})=>`<tr data-i="${i}" tabindex="0" class="${i===selected?'selected':''}"><td>${esc(e.timestamp_et.slice(0,16).replace('T',' '))}</td><td>${esc(e.strategy)} ${esc(e.setup_type)}<br>${esc(e.setup_name)}</td><td>${e.setup_side>0?'Long':'Short'}</td><td>${fmt(e.confidence,3)}</td><td class="${moneyClass(e.net_points_3)}">${fmt(e.net_points_3)}</td><td class="${moneyClass(e.net_points_12)}">${fmt(e.net_points_12)}</td><td class="${moneyClass(e.net_points_20)}">${fmt(e.net_points_20)}</td></tr>`).join('');
  document.getElementById('count').textContent=`${rows.length} / ${data.length} setups`;
  document.querySelectorAll('#events tr').forEach(row=>{row.onclick=()=>selectEvent(Number(row.dataset.i));row.onkeydown=ev=>{if(ev.key==='Enter')selectEvent(Number(row.dataset.i))}});
}
function renderScores(){
  const matchedOnly=document.getElementById('matching').checked;
  const scores=data[selected].scores.filter(s=>!matchedOnly||s.route_match);
  document.querySelector('#scores tbody').innerHTML=scores.map(s=>`<tr><td>${esc(s.model_id)}</td><td>${fmt(s.probability,4)}</td><td>${fmt(s.threshold,4)}</td><td>${s.threshold_pass?'Yes':'No'}</td><td>${s.route_match?'Yes':'No'}</td><td>${esc(local(s.recorded_at_utc))}</td></tr>`).join('');
  if(!scores.length)document.querySelector('#scores tbody').innerHTML='<tr><td colspan="6">No matching saved model. Uncheck the filter to see all scores on this source bar.</td></tr>';
}
function selectEvent(i){
  selected=i;const e=data[i];renderRows();renderScores();
  document.getElementById('event-title').textContent=`${e.timestamp_et.slice(0,16).replace('T',' ')} ET · ${e.strategy} ${e.setup_type} · ${e.setup_side>0?'Long':'Short'} ${e.setup_name}`;
  document.getElementById('event-note').textContent=`Evidence: ${e.provenance}. Saved mark close ${fmt(e.close)}; final canonical close ${fmt(e.canonical_close)}. First same-bar model record: ${local(e.first_model_recorded_at_utc)} ET. ${e.strategy==='ICT'?'Exact original ICT marking time is not retained.':''}`;
  document.getElementById('metrics').innerHTML=[['15m net points',e.net_points_3],['60m net points',e.net_points_12],['100m net points',e.net_points_20],['60m MFE',e.mfe_points_12],['60m MAE',e.mae_points_12]].map(([label,v])=>`<div class="metric"><span>${label}</span><b>${fmt(v)}</b></div>`).join('');
  const traces=[true,false].map(complete=>{const rows=e.bars.filter(b=>b.complete===complete);return {type:'candlestick',name:complete?'Complete bars':'Incomplete bars',x:rows.map(b=>b.time),open:rows.map(b=>b.open),high:rows.map(b=>b.high),low:rows.map(b=>b.low),close:rows.map(b=>b.close),increasing:{line:{color:complete?'#168274':'#969da5'}},decreasing:{line:{color:complete?'#c94d5a':'#969da5'}},showlegend:!complete}});
  const source=e.timestamp_et.slice(0,16).replace('T',' '),entry=local(new Date(new Date(e.timestamp_utc).getTime()+300000).toISOString()).slice(0,16);
  const shapes=[{type:'line',x0:source,x1:source,y0:0,y1:1,yref:'paper',line:{color:'#da8a15',width:2,dash:'dot'}},{type:'line',x0:entry,x1:entry,y0:0,y1:1,yref:'paper',line:{color:'#148276',width:1,dash:'dot'}}];
  Plotly.react('chart',traces,{margin:{t:20,r:20,l:60,b:50},paper_bgcolor:'white',plot_bgcolor:'#fbfcfe',xaxis:{type:'date',rangeslider:{visible:false},title:'New York local time'},yaxis:{title:'ES price',fixedrange:false},shapes,legend:{orientation:'h'},hovermode:'x unified',dragmode:'pan'}, {responsive:true,displaylogo:false,scrollZoom:false});
}
document.getElementById('search').addEventListener('input',renderRows);
document.getElementById('sort').addEventListener('change',renderRows);
document.getElementById('matching').addEventListener('change',renderScores);
selectEvent(0);
