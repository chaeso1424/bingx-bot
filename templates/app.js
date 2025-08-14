async function poll(){
  try{
    const r = await fetch('/status');
    const s = await r.json();

    syncControls(s.running);

    const symEl = document.getElementById('st-symbol');
    if (symEl) {
      const symText = (s.symbol || '-')
        + (s.side ? (' / ' + s.side) : '');
      symEl.textContent = symText;
    }

    document.getElementById('st-avg').textContent =
      (typeof s.avg_price === 'number' && Number.isFinite(s.avg_price))
        ? s.avg_price.toFixed(4) : s.avg_price;

    document.getElementById('st-qty').textContent = s.qty ?? '-';
    document.getElementById('st-tp').textContent = s.tp_order_id || '-';
  }catch(e){
    // 콘솔에서 확인하고 싶으면:
    // console.warn('status poll failed', e);
  }
  setTimeout(poll, 1500);
}
