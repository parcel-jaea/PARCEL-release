(function () {
  var K = 'parcel_api_lang';
  function get() { try { return localStorage.getItem(K); } catch (e) { return null; } }
  function put(v) { try { localStorage.setItem(K, v); } catch (e) {} }
  var groups = Array.prototype.slice.call(document.querySelectorAll('.langtabs'));
  if (!groups.length) return;
  function panes(g) { return Array.prototype.filter.call(g.children, function (e) { return e.classList.contains('langpane'); }); }
  function show(l) {
    groups.forEach(function (g) {
      var ps = panes(g), has = ps.some(function (p) { return p.getAttribute('data-lang') === l; });
      var cur = has ? l : ps[0].getAttribute('data-lang');
      ps.forEach(function (p) { p.hidden = (p.getAttribute('data-lang') !== cur); });
      Array.prototype.forEach.call(g.querySelectorAll('.langtabbar button'), function (b) {
        b.setAttribute('aria-selected', b.getAttribute('data-lang') === cur ? 'true' : 'false');
      });
    });
  }
  function fromHash() {
    var h = decodeURIComponent(location.hash.slice(1));
    if (!h) return null;
    var e = document.getElementById(h);
    var p = e ? e.closest('.langpane') : null;
    return p ? { lang: p.getAttribute('data-lang'), el: e } : null;
  }
  groups.forEach(function (g) {
    g.classList.add('js');
    Array.prototype.forEach.call(g.querySelectorAll('.langtabbar button'), function (b) {
      b.addEventListener('click', function () { var l = b.getAttribute('data-lang'); put(l); show(l); });
    });
  });
  var t = fromHash();
  show(t ? t.lang : (get() || 'cpp'));
  if (t) t.el.scrollIntoView();
  window.addEventListener('hashchange', function () {
    var t = fromHash();
    if (t) { put(t.lang); show(t.lang); t.el.scrollIntoView(); }
  });
})();
