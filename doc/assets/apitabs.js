(function () {
  var K = 'parcel_api_lang', me = document.currentScript, page = (me && me.getAttribute('data-page')) || '';
  function get() { try { return localStorage.getItem(K); } catch (e) { return null; } }
  function put(v) { try { localStorage.setItem(K, v); } catch (e) {} }
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.apilang a'));
  if (page) { put(page); return; }
  function show(l) {
    tabs.forEach(function (t) { t.setAttribute('aria-selected', t.getAttribute('data-apilang') === l ? 'true' : 'false'); });
    Array.prototype.forEach.call(document.querySelectorAll('[data-apilang]'), function (e) {
      var a = e.getAttribute('data-apilang');
      if (!e.closest('.apilang')) e.hidden = (a !== l && a !== 'all');
    });
    var t = tabs.filter(function (x) { return x.getAttribute('data-apilang') === l; })[0];
    if (!t) return;
    var first = t.getAttribute('href'), side = document.querySelector('.sidebar a[href^="' + first + '#"]');
    Array.prototype.forEach.call(document.querySelectorAll('.pagenav .next a[data-langnext]'), function (a) {
      a.setAttribute('href', first);
      if (side) a.textContent = side.textContent.replace(/^\s*[0-9.]+\s*/, '') + ' →';
    });
  }
  var firsts = tabs.map(function (x) { return x.getAttribute('href'); });
  Array.prototype.forEach.call(document.querySelectorAll('.pagenav .next a'), function (a) {
    if (firsts.indexOf(a.getAttribute('href')) >= 0) a.setAttribute('data-langnext', '1');
  });
  var cur = get();
  if (!tabs.some(function (t) { return t.getAttribute('data-apilang') === cur; })) cur = 'cpp';
  show(cur);
  tabs.forEach(function (t) {
    t.addEventListener('click', function (ev) { ev.preventDefault(); var l = t.getAttribute('data-apilang'); put(l); show(l); });
  });
})();
