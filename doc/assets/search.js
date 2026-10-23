(function () {
  var idx = window.PARCEL_SEARCH_INDEX || [];
  var L = window.PARCEL_SEARCH_LABELS;
  var q = document.getElementById('q'), out = document.getElementById('results'), info = document.getElementById('info');
  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }
  function mark(s, terms) {
    var h = esc(s);
    terms.forEach(function (t) {
      var re = new RegExp(esc(t).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
      h = h.replace(re, function (m) { return '<mark>' + m + '</mark>'; });
    });
    return h;
  }
  function run() {
    var s = q.value.trim();
    out.innerHTML = '';
    if (!s) { info.textContent = L.need; return; }
    var terms = s.toLowerCase().split(/\s+/).filter(function (t) { return t.length > 0; });
    var hits = [];
    idx.forEach(function (e) {
      var ht = (e.h + ' ' + e.p).toLowerCase(), tx = e.x.toLowerCase(), score = 0;
      for (var i = 0; i < terms.length; i++) {
        var t = terms[i], inH = ht.indexOf(t) >= 0, pos = tx.indexOf(t);
        if (!inH && pos < 0) return;
        if (inH) score += 10;
        var n = 0, k = pos;
        while (k >= 0 && n < 20) { n++; k = tx.indexOf(t, k + t.length); }
        score += n;
      }
      hits.push([score, e]);
    });
    hits.sort(function (a, b) { return b[0] - a[0]; });
    info.textContent = hits.length ? hits.length + ' ' + L.hits : L.nohit;
    hits.slice(0, 200).forEach(function (h) {
      var e = h[1], tx = e.x, low = tx.toLowerCase(), pos = -1;
      for (var i = 0; i < terms.length && pos < 0; i++) pos = low.indexOf(terms[i]);
      var a = Math.max(0, pos - 60), snip = (a > 0 ? '… ' : '') + tx.substr(a, 200) + (a + 200 < tx.length ? ' …' : '');
      var li = document.createElement('li');
      li.innerHTML = '<a href="' + esc(e.u) + '">' + mark(e.h || e.p, terms) + '</a>' +
        '<div class="where">' + esc(e.d + ' › ' + e.p) + '</div>' +
        '<div class="snippet">' + mark(snip, terms) + '</div>';
      out.appendChild(li);
    });
  }
  document.getElementById('f').addEventListener('submit', function (ev) { ev.preventDefault(); run(); });
  q.addEventListener('input', function () { if (q.value.trim().length >= 2 || !q.value.trim()) run(); });
  var m = /[?&]q=([^&]*)/.exec(location.search);
  if (m) { q.value = decodeURIComponent(m[1].replace(/\+/g, ' ')); run(); }
})();
