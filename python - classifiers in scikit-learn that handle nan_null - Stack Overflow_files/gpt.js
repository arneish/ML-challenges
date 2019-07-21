(function(E){var window=this;var aa=function(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}},ba=function(a){var b="undefined"!=typeof Symbol&&Symbol.iterator&&a[Symbol.iterator];return b?b.call(a):{next:aa(a)}},ca=function(a){if(!(a instanceof Array)){a=ba(a);for(var b,c=[];!(b=a.next()).done;)c.push(b.value);a=c}return a},g=this||self,k=function(a){return"string"==typeof a},da=function(a){return"number"==typeof a},ha=function(){if(null===ea)a:{var a=g.document;if((a=a.querySelector&&a.querySelector("script[nonce]"))&&(a=a.nonce||a.getAttribute("nonce"))&&fa.test(a)){ea=a;break a}ea=""}return ea},fa=/^[\w+/_-]+[=]{0,2}$/,ea=null,ia=function(a){a=a.split(".");for(var b=g,c=0;c<a.length;c++)if(b=b[a[c]],null==b)return null;return b},ja=function(){},l=function(a){a.m=void 0;a.b=function(){return a.m?a.m:a.m=new a}},n=function(a){var b=typeof a;if("object"==b)if(a){if(a instanceof Array)return"array";if(a instanceof Object)return b;var c=Object.prototype.toString.call(a);if("[object Window]"==c)return"object";if("[object Array]"==c||"number"==typeof a.length&&"undefined"!=typeof a.splice&&"undefined"!=typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("splice"))return"array";if("[object Function]"==c||"undefined"!=typeof a.call&&"undefined"!=typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("call"))return"function"}else return"null";else if("function"==b&&"undefined"==typeof a.call)return"object";return b},ka="closure_uid_"+(1E9*Math.random()>>>0),la=0,ma=function(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var d=c.slice();d.push.apply(d,arguments);return a.apply(this,d)}},na=function(a,b){for(var c in b)a[c]=b[c]},p=function(a,b){function c(){}c.prototype=b.prototype;a.I=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.G=function(d,e,f){for(var h=Array(arguments.length-2),m=2;m<arguments.length;m++)h[m-2]=arguments[m];return b.prototype[e].apply(d,h)}};var q=function(a,b){for(var c=a.length,d=k(a)?a.split(""):a,e=0;e<c;e++)e in d&&b.call(void 0,d[e],e,a)},oa=function(a,b){for(var c=a.length,d=[],e=0,f=k(a)?a.split(""):a,h=0;h<c;h++)if(h in f){var m=f[h];b.call(void 0,m,h,a)&&(d[e++]=m)}return d},pa=function(a,b){for(var c=a.length,d=Array(c),e=k(a)?a.split(""):a,f=0;f<c;f++)f in e&&(d[f]=b.call(void 0,e[f],f,a));return d},qa=function(a,b){a:{for(var c=a.length,d=k(a)?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){b=e;break a}b=-1}return 0>b?null:k(a)?a.charAt(b):a[b]},ra=function(a,b){a:{for(var c=k(a)?a.split(""):a,d=a.length-1;0<=d;d--)if(d in c&&b.call(void 0,c[d],d,a)){b=d;break a}b=-1}return 0>b?null:k(a)?a.charAt(b):a[b]},sa=function(a,b){a:if(k(a))a=k(b)&&1==b.length?a.indexOf(b,0):-1;else{for(var c=0;c<a.length;c++)if(c in a&&a[c]===b){a=c;break a}a=-1}return 0<=a};var ta=function(a){var b=!1,c;return function(){b||(c=a(),b=!0);return c}};var ua=function(a,b){return null!==a&&b in a};var r=function(){this.a="";this.h=va};r.prototype.f=!0;r.prototype.c=function(){return this.a.toString()};var wa=function(a){return a instanceof r&&a.constructor===r&&a.h===va?a.a:"type_error:TrustedResourceUrl"},va={};var xa=function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]},t=function(a,b){var c=0;a=xa(String(a)).split(".");b=xa(String(b)).split(".");for(var d=Math.max(a.length,b.length),e=0;0==c&&e<d;e++){var f=a[e]||"",h=b[e]||"";do{f=/(\d*)(\D*)(.*)/.exec(f)||["","","",""];h=/(\d*)(\D*)(.*)/.exec(h)||["","","",""];if(0==f[0].length&&0==h[0].length)break;c=ya(0==f[1].length?0:parseInt(f[1],10),0==h[1].length?0:parseInt(h[1],10))||ya(0==f[2].length,0==h[2].length)||ya(f[2],h[2]);f=f[3];h=h[3]}while(0==c)}return c},ya=function(a,b){return a<b?-1:a>b?1:0};var u=function(){this.a="";this.h=za};u.prototype.f=!0;u.prototype.c=function(){return this.a.toString()};var Aa=function(a){return a instanceof u&&a.constructor===u&&a.h===za?a.a:"type_error:SafeUrl"},Ba=/^(?:(?:https?|mailto|ftp):|[^:/?#]*(?:[/?#]|$))/i,za={},Ca=function(a){var b=new u;b.a=a;return b};Ca("about:blank");var v;a:{var Ea=g.navigator;if(Ea){var Fa=Ea.userAgent;if(Fa){v=Fa;break a}}v=""}var w=function(a){return-1!=v.indexOf(a)},Ga=function(a){for(var b=/(\w[\w ]+)\/([^\s]+)\s*(?:\((.*?)\))?/g,c=[],d;d=b.exec(a);)c.push([d[1],d[2],d[3]||void 0]);return c};var Ha=function(){return(w("Chrome")||w("CriOS"))&&!w("Edge")},Ja=function(){function a(e){e=qa(e,d);return c[e]||""}var b=v;if(w("Trident")||w("MSIE"))return Ia(b);b=Ga(b);var c={};q(b,function(e){c[e[0]]=e[1]});var d=ma(ua,c);return w("Opera")?a(["Version","Opera"]):w("Edge")?a(["Edge"]):w("Edg/")?a(["Edg"]):Ha()?a(["Chrome","CriOS"]):(b=b[2])&&b[1]||""},Ia=function(a){var b=/rv: *([\d\.]*)/.exec(a);if(b&&b[1])return b[1];b="";var c=/MSIE +([\d\.]+)/.exec(a);if(c&&c[1])if(a=/Trident\/(\d.\d)/.exec(a),"7.0"==c[1])if(a&&a[1])switch(a[1]){case "4.0":b="8.0";break;case "5.0":b="9.0";break;case "6.0":b="10.0";break;case "7.0":b="11.0"}else b="7.0";else b=c[1];return b};var Ka=function(a,b){a.src=wa(b);(b=ha())&&a.setAttribute("nonce",b)};var x=function(a){x[" "](a);return a};x[" "]=ja;var z=function(){},La="function"==typeof Uint8Array,B=function(a,b,c,d){a.a=null;b||(b=[]);a.H=void 0;a.f=-1;a.g=b;a:{if(b=a.g.length){--b;var e=a.g[b];if(!(null===e||"object"!=typeof e||"array"==n(e)||La&&e instanceof Uint8Array)){a.h=b-a.f;a.c=e;break a}}a.h=Number.MAX_VALUE}a.v={};if(c)for(b=0;b<c.length;b++)e=c[b],e<a.h?(e+=a.f,a.g[e]=a.g[e]||A):(Ma(a),a.c[e]=a.c[e]||A);if(d&&d.length)for(b=0;b<d.length;b++)Na(a,d[b])},A=[],Ma=function(a){var b=a.h+a.f;a.g[b]||(a.c=a.g[b]={})},C=function(a,b){if(b<a.h){b+=a.f;var c=a.g[b];return c===A?a.g[b]=[]:c}if(a.c)return c=a.c[b],c===A?a.c[b]=[]:c},D=function(a,b,c){a=C(a,b);return null==a?c:a},Oa=function(a,b){a=C(a,b);a=null==a?a:+a;return null==a?0:a},Pa=function(a,b,c){b<a.h?a.g[b+a.f]=c:(Ma(a),a.c[b]=c)},Na=function(a,b){for(var c,d,e=0;e<b.length;e++){var f=b[e],h=C(a,f);null!=h&&(c=f,d=h,Pa(a,f,void 0))}return c?(Pa(a,c,d),c):0},F=function(a,b,c){a.a||(a.a={});if(!a.a[c]){var d=C(a,c);d&&(a.a[c]=new b(d))}return a.a[c]},G=function(a,b,c){a.a||(a.a={});if(!a.a[c]){for(var d=C(a,c),e=[],f=0;f<d.length;f++)e[f]=new b(d[f]);a.a[c]=e}b=a.a[c];b==A&&(b=a.a[c]=[]);return b},Qa=function(a,b,c){a.a||(a.a={});c=c||[];for(var d=[],e=0;e<c.length;e++)d[e]=c[e].g;a.a[b]=c;Pa(a,b,d)};var Sa=function(a){Ra();var b=new r;b.a=a;return b},Ra=ja;var Ta=function(a,b){a=[a];for(var c=b.length-1;0<=c;--c)a.push(typeof b[c],b[c]);return a.join("\x0B")};var Ua=/^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#([\s\S]*))?$/;var Ya=function(a,b){if(!Va()&&!Wa()){var c=Math.random();if(c<b)return c=Xa(g),a[Math.floor(c*a.length)]}return null},Xa=function(a){if(!a.crypto)return Math.random();try{var b=new Uint32Array(1);a.crypto.getRandomValues(b);return b[0]/65536/65536}catch(c){return Math.random()}},Wa=ta(function(){return-1!=v.indexOf("Google Web Preview")||1E-4>Math.random()}),Va=ta(function(){return-1!=v.indexOf("MSIE")}),Za=/^(-?[0-9.]{1,30})$/,$a=function(a,b){return Za.test(a)&&(a=Number(a),!isNaN(a))?a:void 0==b?null:b},ab=function(){try{return ha()}catch(a){}};var bb=function(a){var b=window,c=-1;a="google_experiment_mod"+(void 0===a?"":a);try{b.localStorage&&(c=parseInt(b.localStorage.getItem(a),10))}catch(d){return null}if(0<=c&&1E3>c)return c;if(Wa())return null;c=Math.floor(1E3*Xa(b));try{if(b.localStorage)return b.localStorage.setItem(a,""+c),c}catch(d){}return null};var cb=function(a,b){var c=void 0===c?{}:c;this.error=a;this.context=b.context;this.msg=b.message||"";this.id=b.id||"jserror";this.meta=c};var db=null,eb=function(){if(null===db){db="";try{var a="";try{a=g.top.location.hash}catch(c){a=g.location.hash}if(a){var b=a.match(/\bdeid=([\d,]+)/);db=b?b[1]:""}}catch(c){}}return db};var H=function(a){B(this,a,fb,gb)};p(H,z);var fb=[2,8],gb=[[3,4,5],[6,7]];var hb=function(a){return null!=a?!a:a},ib=function(a,b){for(var c=!1,d=0;d<a.length;d++){var e=a[d].call();if(e==b)return e;null==e&&(c=!0)}if(!c)return!b},kb=function(a,b){var c=G(a,H,2);if(!c.length)return jb(a,b);a=D(a,1,0);if(1==a)return hb(kb(c[0],b));c=pa(c,function(d){return function(){return kb(d,b)}});switch(a){case 2:return ib(c,!1);case 3:return ib(c,!0)}},jb=function(a,b){var c=Na(a,gb[0]);a:{switch(c){case 3:var d=D(a,3,0);break a;case 4:d=D(a,4,0);break a;case 5:d=D(a,5,0);break a}d=void 0}if(d&&(b=(b=b[c])&&b[d])){try{var e=b.apply(null,C(a,8))}catch(f){return}b=D(a,1,0);if(4==b)return!!e;d=null!=e;if(5==b)return d;a:{switch(c){case 4:a=Oa(a,6);break a;case 5:a=D(a,7,"");break a}a=void 0}if(null!=a){if(6==b)return e===a;if(9==b)return 0==t(e,a);if(d)switch(b){case 7:return e<a;case 8:return e>a;case 12:return(new RegExp(a)).test(e);case 10:return-1==t(e,a);case 11:return 1==t(e,a)}}}},lb=function(a,b){return!a||!(!b||!kb(a,b))};var nb=function(a){B(this,a,mb,null)};p(nb,z);var mb=[4];var I=function(a){B(this,a,ob,pb)};p(I,z);var qb=function(a){B(this,a,null,null)};p(qb,z);var ob=[5],pb=[[1,2,3,6]];var J=function(){var a={};this.a=(a[3]={},a[4]={},a[5]={},a)},rb=function(a,b){a.a=b};l(J);var sb=function(a,b){switch(b){case 1:return D(a,1,0);case 2:return D(a,2,0);case 3:return D(a,3,0);case 6:return D(a,6,0);default:return null}},tb=function(a,b){if(!a)return null;switch(b){case 1:return a=C(a,1),a=null==a?a:!!a,null==a?!1:a;case 2:return Oa(a,2);case 3:return D(a,3,"");case 6:return C(a,4);default:return null}},ub=ta(function(){var a="";try{a=g.top.location.hash}catch(c){a=g.location.hash}var b={};if(a=(a=/\bdflags=({.*})(&|$)/.exec(a))&&a[1])try{b=JSON.parse(decodeURIComponent(a))}catch(c){}return b}),wb=function(a,b,c){var d=ub();if(d[a]&&null!=d[a][b])return d[a][b];b=K.b().a[a][b];if(!b)return c;b=new I(b);b=vb(b);a=tb(b,a);return null!=a?a:c},vb=function(a){var b=J.b().a;if(b){var c=ra(G(a,qb,5),function(d){return lb(F(d,H,1),b)});if(c)return F(c,nb,2)}return F(a,nb,4)},K=function(){var a={};this.a=(a[1]={},a[2]={},a[3]={},a[6]={},a)};l(K);var xb=function(a,b){return wb(1,a,void 0===b?!1:b)},yb=function(a,b){return wb(2,a,void 0===b?0:b)},zb=function(a,b){return wb(3,a,void 0===b?"":b)},Ab=function(a,b){b=void 0===b?[]:b;return wb(6,a,b)},Bb=function(a){var b=K.b().a;q(a,function(c){var d=Na(c,pb[0]),e=sb(c,d);e&&(b[d][e]=c.g)})},Cb=function(a){var b=K.b().a;q(a,function(c){var d=new I(c),e=Na(d,pb[0]);(d=sb(d,e))&&(b[e][d]||(b[e][d]=c))})};var L=function(a){this.a=a},Db=new L(1),Eb=new L(2),Fb=new L(3),Gb=new L(4),Hb=new L(5),Ib=new L(6),Jb=new L(7),Kb=new L(8),Lb=new L(9),Mb=new L(10),Nb=new L(11),Ob=new L(12),Pb=new L(13),Qb=new L(14),M=function(a,b,c){c.hasOwnProperty(a.a)||Object.defineProperty(c,String(a.a),{value:b})},N=function(a,b){return b[a.a]||function(){}},Rb=function(a){M(Hb,xb,a);M(Ib,yb,a);M(Jb,zb,a);M(Kb,Ab,a);M(Pb,Cb,a)},Sb=function(a){M(Gb,function(b){J.b().a=b},a);M(Lb,function(b,c){var d=J.b();d.a[3][b]||(d.a[3][b]=c)},a);M(Mb,function(b,c){var d=J.b();d.a[4][b]||(d.a[4][b]=c)},a);M(Nb,function(b,c){var d=J.b();d.a[5][b]||(d.a[5][b]=c)},a);M(Qb,function(b){for(var c=J.b(),d=ba([3,4,5]),e=d.next();!e.done;e=d.next())e=e.value,na(c.a[e],b[e])},a)},Tb=function(a){a.hasOwnProperty("init-done")||Object.defineProperty(a,"init-done",{value:!0})};var Ub=function(){var a=void 0===a?g:a;return a.ggeac||(a.ggeac={})};var O=function(){this.a=function(){return[]};this.c=function(){return[]}};l(O);var Wb=function(a){B(this,a,Vb,null)};p(Wb,z);var Vb=[2];Wb.prototype.getId=function(){return D(this,1,0)};Wb.prototype.i=function(){return D(this,7,0)};var Yb=function(a){B(this,a,Xb,null)};p(Yb,z);var Xb=[2];Yb.prototype.i=function(){return D(this,5,0)};var $b=function(a){B(this,a,Zb,null)};p($b,z);var P=function(a){B(this,a,ac,null)};p(P,z);var Zb=[1,2],ac=[2];P.prototype.i=function(){return D(this,1,0)};var bc=[1,12,13],cc=function(a,b){var c=this,d=void 0===b?{}:b;b=void 0===d.u?!1:d.u;var e=void 0===d.A?{}:d.A;d=void 0===d.D?[]:d.D;this.a=a;this.v=b;this.f=e;this.h=d;this.c={};(a=eb())&&q(a.split(",")||[],function(f){(f=parseInt(f,10))&&(c.c[f]=!0)})},gc=function(a,b){var c=[],d=dc(a.a,b);d.length&&(9!==b&&(a.a=ec(a.a,b)),q(d,function(e){if(e=fc(a,e)){var f=e.getId();c.push(f);a.h.push(f);(e=G(e,I,2))&&Bb(e)}}));return c},hc=function(a,b){a.a.push.apply(a.a,ca(oa(pa(b,function(c){return new P(c)}),function(c){return!sa(bc,c.i())})))},fc=function(a,b){var c=J.b().a;if(!lb(F(b,H,3),c))return null;var d=G(b,Wb,2),e=c?oa(d,function(h){return lb(F(h,H,3),c)}):d,f=e.length;if(!f)return null;d=D(b,4,0);b=f*D(b,1,0);if(!d)return ic(a,e,b/1E3);f=null!=a.f[d]?a.f[d]:1E3;if(0>=f)return null;e=ic(a,e,b/f);a.f[d]=e?0:f-b;return e},ic=function(a,b,c){var d=a.c,e=qa(b,function(f){return!!d[f.getId()]});return e?e:a.v?null:Ya(b,c)},jc=function(a,b){M(Db,function(c){a.c[c]=!0},b);M(Eb,function(c){return gc(a,c)},b);M(Fb,function(){return a.h},b);M(Ob,function(c){return hc(a,c)},b)},dc=function(a,b){return(a=qa(a,function(c){return c.i()==b}))&&G(a,Yb,2)||[]},ec=function(a,b){return oa(a,function(c){return c.i()!=b})};var Q=function(){this.a=function(){return!1};this.c=function(){return 0};this.f=function(){return""}};l(Q);var kc=function(a){var b=void 0===b?!1:b;return Q.b().a(a,b)},lc=function(a,b){b=void 0===b?"":b;return Q.b().f(a,b)};var mc=function(){};l(mc);var oc=function(a,b){var c={u:R(211),A:R(227),D:R(226)};var d=void 0===d?Ub():d;d.hasOwnProperty("init-done")?(N(Ob,d)(pa(G(a,P,2),function(e){return e.g})),N(Pb,d)(pa(G(a,I,1),function(e){return e.g})),b&&N(Qb,d)(b),nc(d)):(jc(new cc(G(a,P,2),c),d),Rb(d),Sb(d),Tb(d),nc(d),Bb(G(a,I,1)),b&&rb(J.b(),b))},nc=function(a){var b=a=void 0===a?Ub():a,c=O.b();c.a=N(Eb,b);c.c=N(Fb,b);b=Q.b();b.a=N(Hb,a);b.c=N(Ib,a);b.f=N(Jb,a);mc.b()};var pc=ta(function(){var a=g.navigator&&g.navigator.userAgent||"";a=a.toLowerCase();return-1!=a.indexOf("firefox/")||-1!=a.indexOf("chrome/")||-1!=a.indexOf("opr/")}),qc=function(a,b,c,d,e){d=void 0===d?"":d;var f=a.createElement("link");try{f.rel=c;if(-1!=c.toLowerCase().indexOf("stylesheet"))var h=wa(b).toString();else{if(b instanceof r)var m=wa(b).toString();else{if(b instanceof u)var y=Aa(b);else{if(b instanceof u)var X=b;else b="object"==typeof b&&b.f?b.c():String(b),Ba.test(b)||(b="about:invalid#zClosurez"),X=Ca(b);y=Aa(X)}m=y}h=m}f.href=h}catch(Da){return}d&&"preload"==c&&(f.as=d);e&&f.setAttribute("nonce",e);if(a=a.getElementsByTagName("head")[0])try{a.appendChild(f)}catch(Da){}};var rc=/^\.google\.(com?\.)?[a-z]{2,3}$/,sc=/\.(cn|com\.bi|do|sl|ba|by|ma|am)$/,tc=function(a){return rc.test(a)&&!sc.test(a)},uc=function(a){return a.replace(/[\W]/g,function(b){return"&#"+b.charCodeAt()+";"})},S=g,vc=function(a,b){a="https://adservice"+(b+"/adsid/integrator."+a);b=["domain="+encodeURIComponent(g.location.hostname)];T[3]>=+new Date&&b.push("adsid="+encodeURIComponent(T[1]));return a+"?"+b.join("&")},T,U,wc=function(){S=g;T=S.googleToken=S.googleToken||{};var a=+new Date;T[1]&&T[3]>a&&0<T[2]||(T[1]="",T[2]=-1,T[3]=-1,T[4]="",T[6]="");U=S.googleIMState=S.googleIMState||{};tc(U[1])||(U[1]=".google.com");"array"==n(U[5])||(U[5]=[]);"boolean"==typeof U[6]||(U[6]=!1);"array"==n(U[7])||(U[7]=[]);da(U[8])||(U[8]=0)},xc=function(a){try{a()}catch(b){g.setTimeout(function(){throw b;},0)}},zc=function(a){"complete"==g.document.readyState||"loaded"==g.document.readyState||g.document.currentScript&&g.document.currentScript.async?yc(3):a()},Ac=0,V={j:function(){return 0<U[8]},o:function(){U[8]++},B:function(){0<U[8]&&U[8]--},C:function(){U[8]=0},l:function(){},F:function(){return!1},w:function(){return U[5]},s:xc},W={j:function(){return U[6]},o:function(){U[6]=!0},B:function(){U[6]=!1},C:function(){U[6]=!1},l:function(){},F:function(){return".google.com"!=U[1]&&2<++Ac},w:function(){return U[7]},s:function(a){zc(function(){xc(a)})}},yc=function(a){if(1E-5>Math.random()){g.google_image_requests||(g.google_image_requests=[]);var b=g.document.createElement("img");b.src="https://pagead2.googlesyndication.com/pagead/gen_204?id=imerr&err="+a;g.google_image_requests.push(b)}};V.l=function(){if(!V.j()){var a=g.document,b=function(e){e=vc("js",e);var f=ab();qc(a,e,"preload","script",f);f=a.createElement("script");f.type="text/javascript";f.onerror=function(){return g.processGoogleToken({},2)};e=Sa(e);Ka(f,e);try{(a.head||a.body||a.documentElement).appendChild(f),V.o()}catch(h){}},c=U[1];b(c);".google.com"!=c&&b(".google.com");b={};var d=(b.newToken="FBT",b);g.setTimeout(function(){return g.processGoogleToken(d,1)},1E3)}};W.l=function(){if(!W.j()){var a=g.document,b=vc("sync.js",U[1]);qc(a,b,"preload","script");b=uc(b);var c=x("script"),d="",e=ab();e&&(d='nonce="'+uc(e)+'"');var f="<"+c+' src="'+b+'" '+d+"></"+c+"><"+(c+" "+d+'>processGoogleTokenSync({"newToken":"FBS"},5);</'+c+">");zc(function(){a.write(f);W.o()})}};var Bc=function(a){wc();T[3]>=+new Date&&T[2]>=+new Date||a.l()},Dc=function(){g.processGoogleToken=g.processGoogleToken||function(a,b){return Cc(V,a,b)};Bc(V)},Ec=function(){g.processGoogleTokenSync=g.processGoogleTokenSync||function(a,b){return Cc(W,a,b)};Bc(W)},Cc=function(a,b,c){b=void 0===b?{}:b;c=void 0===c?0:c;var d=b.newToken||"",e="NT"==d,f=parseInt(b.freshLifetimeSecs||"",10),h=parseInt(b.validLifetimeSecs||"",10),m=b["1p_jar"]||"";b=b.pucrd||"";wc();1==c?a.C():a.B();if(!d&&a.F())tc(".google.com")&&(U[1]=".google.com"),a.l();else{var y=S.googleToken=S.googleToken||{},X=0==c&&d&&k(d)&&!e&&da(f)&&0<f&&da(h)&&0<h&&k(m);e=e&&!a.j()&&(!(T[3]>=+new Date)||"NT"==T[1]);var Da=!(T[3]>=+new Date)&&0!=c;if(X||e||Da)e=+new Date,f=e+1E3*f,h=e+1E3*h,yc(c),y[5]=c,y[1]=d,y[2]=f,y[3]=h,y[4]=m,y[6]=b,wc();if(X||!a.j()){c=a.w();for(d=0;d<c.length;d++)a.s(c[d]);c.length=0}}};var Fc=function(){this.a=null;this.f=this.c},Gc=function(a,b){a.a=b};Fc.prototype.c=function(a,b,c,d,e){if(Math.random()>(void 0===c?.01:c))return!1;b.error&&b.meta&&b.id||(b=new cb(b,{context:a,id:void 0===e?"gpt_exception":e}));if(d||this.a)b.meta={},this.a&&this.a(b.meta),d&&d(b.meta);g.google_js_errors=g.google_js_errors||[];g.google_js_errors.push(b);g.error_rep_loaded||(b=g.document,a=b.createElement("script"),Ka(a,Sa(g.location.protocol+"//pagead2.googlesyndication.com/pagead/js/err_rep.js")),(b=b.getElementsByTagName("script")[0])&&b.parentNode&&b.parentNode.insertBefore(a,b),g.error_rep_loaded=!0);return!1};var Hc=function(a,b){try{b()}catch(c){if(!a.f(420,c,.01,void 0,"gpt_exception"))throw c;}};var Ic=function(a){if(!a.google_ltobserver){var b=new a.PerformanceObserver(function(c){var d=a.google_lt_queue=a.google_lt_queue||[];q(c.getEntries(),function(e){return d.push(e)})});b.observe({entryTypes:["longtask"]});a.google_ltobserver=b}};var Jc=function(a){var b=a;b=void 0===b?g:b;if(b=(b=b.performance)&&b.now?b.now():null)b={label:"1",type:9,value:b},a=a.google_js_reporting_queue=a.google_js_reporting_queue||[],2048>a.length&&a.push(b)};var Kc=function(){return g.googletag||(g.googletag={})},Lc=function(a,b){var c=Kc();c.hasOwnProperty(a)||(c[a]=b)};var Y={173:"pubads.g.doubleclick.net",174:"securepubads.g.doubleclick.net",7:.02,13:1500,23:.001,24:200,37:.01,38:.001,58:1,76:"",150:".google.co.in",211:!1,152:[],172:null,191:"001906272103440",192:"021906111828200",190:"011906111828200",245:{},180:null,230:{},246:[],227:{},226:[],241:{},220:!1,228:"//www.googletagservices.com/pubconsole/",242:!1,244:!1,243:-1};Y[6]=function(a,b){b=void 0===b?!0:b;try{for(var c=null;c!=a;c=a,a=a.parent)switch(a.location.protocol){case "https:":return!0;case "file:":return b;case "http:":return!1}}catch(d){}return!0}(window);Y[49]=(new Date).getTime();Y[36]=/^true$/.test("false");Y[46]=/^true$/.test("false");Y[148]=/^true$/.test("false");Y[221]=/^true$/.test("");Y[204]=$a("{{MOD}}",-1);var Mc=function(){na(this,Y)};l(Mc);var R=function(a){return Mc.b()[a]},Z=function(a,b){Mc.b()[a]=b},Nc=Kc(),Oc=Mc.b();na(Oc,Nc._vars_);Nc._vars_=Oc;var Pc=function(){return R(36)};var Qc=function(a,b){var c=b||Ta;return function(){var d=this||g;d=d.closure_memoize_cache_||(d.closure_memoize_cache_={});var e=c(a[ka]||(a[ka]=++la),arguments);return d.hasOwnProperty(e)?d[e]:d[e]=a.apply(this,arguments)}}(function(a){return a&&a.src?/^(?:https?:)?\/\/(?:www\.googletagservices\.com|securepubads\.g\.doubleclick\.net)\/tag\/js\/gpt(?:_[a-z]+)*\.js/.test(a.src)?0:1:2},function(a,b){return a+"\x0B"+(b[0]&&b[0].src)}),Rc=function(){return 0===Qc(R(172))};var Sc=function(){return $a("1")||0},Tc=function(){return"2019070801"};Lc("getVersion",Tc);var Uc=function(){var a={};this[3]=(a[8]=function(b){return!!ia(b)},a[3]=Rc,a[2]=Pc,a[17]=function(b){for(var c=[],d=0;d<arguments.length;++d)c[d]=arguments[d];d=String;var e=void 0===e?window:e;if(e=(e=e.location.href.match(Ua)[3]||null)?decodeURI(e):e){var f=e.length;if(0==f)e=0;else{for(var h=305419896,m=0;m<f;m++)h^=(h<<5)+(h>>2)+e.charCodeAt(m)&4294967295;e=0<h?h:4294967296+h}}else e=null;return sa(c,d(e))},a[9]=function(b){b=ia(b);var c;if(c="function"==n(b))b=b&&b.toString&&b.toString(),c=k(b)&&-1!=b.indexOf("[native code]");return c},a[10]=function(){return window==window.top},a[16]=function(){return Ha()&&0<=t(Ja(),72)||w("Edge")&&0<=t(Ja(),18)||(w("Firefox")||w("FxiOS"))&&0<=t(Ja(),65)||w("Safari")&&!(Ha()||w("Coast")||w("Opera")||w("Edge")||w("Edg/")||w("OPR")||w("Firefox")||w("FxiOS")||w("Silk")||w("Android"))&&0<=t(Ja(),12)},a);a={};this[4]=(a[1]=function(){return R(204)},a[4]=Sc,a[5]=function(b){b=bb(void 0===b?"":b);return null!=b?b:void 0},a[6]=function(b){b=ia(b);return da(b)?b:void 0},a);a={};this[5]=(a[2]=function(){return window.location.href},a[3]=function(){try{return window.top.location.hash}catch(b){return""}},a[4]=function(b){b=ia(b);return k(b)?b:void 0},a)};l(Uc);var Vc=[],Xc=function(a){a=Wc(new $b(R(246)),new $b(a||Vc));var b=Uc.b();b[3][6]=function(c){return sa(O.b().c(),parseInt(c,10))};Z(241,b);oc(a,b);Z(230,K.b().a)},Wc=function(a,b){if(!G(a,I,1).length&&G(b,I,1).length){var c=G(b,I,1);Qa(a,1,c)}!G(a,P,2).length&&G(b,P,2).length&&(b=G(b,P,2),Qa(a,2,b));return a};x("partner.googleadservices.com");var Yc=x("www.googletagservices.com"),Zc=function(){return!R(46)||R(6)||kc(152)?"https://securepubads.g.doubleclick.net":"http://pubads.g.doubleclick.net"},$c=function(a){var b=a.currentScript;return"complete"!=a.readyState&&"loaded"!=a.readyState&&!(b&&b.async)},ad=function(){var a=R(76);if(a)return a;a=Zc();var b=lc(4,"/gpt/pubads_impl_"),c;var d="";if(R(148))try{var e="";try{e=g.top.location.hash}catch(h){e=g.location.hash}e&&(d=(c=e.match(/\bgptv=(\d+)/))?c[1]:"")}catch(h){}if(!(c=d)){var f=void 0===f?0:f;c=Q.b().c(12,f)}c=c||Tc();f=lc(5);a=a+b+c+".js";f&&(a+="?"+f);Z(76,a);return a},bd=function(a,b){var c;if(!(c=a.currentScript))a:{if(a=a.scripts)for(c=0;c<a.length;c++){var d=a[c];if(-1<d.src.indexOf(Yc+"/tag/js/gpt")){c=d;break a}}c=null}Z(172,c);new Xc(b);O.b().a(5);O.b().a(12);b=R(150);wc();tc(b)&&(U[1]=b)},cd=function(){return navigator.getBattery?navigator.getBattery().then(function(a){Z(243,a.level);Z(244,a.charging);Z(242,!0)}):null},dd=function(a,b,c){var d=Kc();a=a||d.fifWin||window;b=b||a.document;Lc("cmd",[]);if(d.evalScripts)d.evalScripts();else{bd(b,c);a.PerformanceObserver&&a.PerformanceLongTaskTiming&&Ic(a);Jc(a);a=ad();if($c(b)){c="gpt-impl-"+Math.random();try{var e='<script id="'+c+'" src="'+a+'">\x3c/script>';kc(17)&&pc()&&(e+='<link rel="preconnect" href="'+Zc()+'">');b.write(e)}catch(f){}b.getElementById(c)&&(d._loadStarted_=!0,Z(220,!1),Ec())}d._loadStarted_||(Dc(),kc(16)&&qc(b,a,"preload","script"),c=b.createElement("script"),c.src=a,c.async=!0,(b.head||b.body||b.documentElement).appendChild(c),kc(17)&&pc()&&qc(b,Zc(),"preconnect"),Z(220,!0),d._loadStarted_=!0);(b=cd())&&b.catch(function(f){var h=new Fc;Gc(h,function(m){m.methodId=501});h.c(501,f)})}};var ed;a:{try{if("array"==n(E)){ed=E;break a}}catch(a){}ed=[]}(function(a,b,c){var d=new Fc;Gc(d,function(e){e.methodId=420});Hc(d,function(){return dd(a,b,c)})})(void 0,void 0,ed);}).call(this.googletag&&googletag.fifWin?googletag.fifWin.parent:this,[[[null,13,null,[null,1]],[146,null,null,[1]],[null,7,null,[null,0.1]],[167,null,null,[1]],[118,null,null,[1]],[20,null,null,[],[[[1,[[4,null,1]]],[1]]]],[90,null,null,[1]],[null,null,8,[null,null,"/pagead/js/rum.js"]],[152,null,null,[1]],[151,null,null,[1]],[158,null,null,[1]],[8,null,null,[1]],[55,null,null,[1]],[null,8,null,[null,-1]],[null,1,null,[null,4096],[[[4,null,14],[null,8192]],[[4,null,15,null,null,null,null,["7646"]],[null,16384]]]],[null,null,9,[null,null,"https://securepubads.g.doubleclick.net/pagead/js/rum.js"]],[45,null,null,[]],[null,null,2,[null,null,"1-0-35"]],[161,null,null,[1]]],[[null,[[null,[[676982416]]],[null,[[676982601],[676982602],[676982605]]],[null,[[676982612],[676982613]]],[null,[[676982665]]],[null,[[676982678]]],[null,[[676982680]]],[null,[[676982682]]]]],[4,[[null,[[676982417]]],[null,[[676982603],[676982604]]],[null,[[676982661],[676982662],[676982663]]],[null,[[676982666],[676982667],[676982669],[676982670]]],[null,[[676982672],[676982673],[676982674],[676982675],[676982676],[676982677]]],[null,[[676982681]]]]],[12,[[1,[[21064123],[21064124]]]]],[null,[[null,[[21063438],[21063439]]],[null,[[21063445],[21063446]]],[null,[[21064058]]],[null,[[21064213],[21064214]]],[null,[[676982416]]],[null,[[676982601],[676982602],[676982605]]],[null,[[676982612],[676982613]]],[null,[[676982665]]],[null,[[676982678]]],[null,[[676982680]]],[null,[[676982682]]]]],[2,[[1000,[[21063912]],[2,[[4,null,6,null,null,null,null,["21063910"]],[2,[[4,null,9,null,null,null,null,["XMLHttpRequest"]],[4,null,9,null,null,null,null,["XMLHttpRequest.prototype.open"]],[4,null,9,null,null,null,null,["XMLHttpRequest.prototype.send"]]]]]]],[1000,[[21063913]],[2,[[4,null,6,null,null,null,null,["21063911"]],[2,[[4,null,9,null,null,null,null,["XMLHttpRequest"]],[4,null,9,null,null,null,null,["XMLHttpRequest.prototype.open"]],[4,null,9,null,null,null,null,["XMLHttpRequest.prototype.send"]]]]]]],[1000,[[21064076]],[2,[[4,null,6,null,null,null,null,["21064055"]],[2,[[4,null,9,null,null,null,null,["XMLHttpRequest"]],[4,null,9,null,null,null,null,["XMLHttpRequest.prototype.open"]],[4,null,9,null,null,null,null,["XMLHttpRequest.prototype.send"]]]]]]],[1000,[[21064077]],[2,[[4,null,6,null,null,null,null,["21064056"]],[2,[[4,null,9,null,null,null,null,["XMLHttpRequest"]],[4,null,9,null,null,null,null,["XMLHttpRequest.prototype.open"]],[4,null,9,null,null,null,null,["XMLHttpRequest.prototype.send"]]]]]]]]],[3,[[null,[[1337,[[82,null,null,[1]],[null,null,8,[null,null,"/pagead/js/rum_debug.js"]]]]]],[5,[[20194812,[[20,null,null,[1]]]],[20194813]],null,3],[500,[[21060610],[21060611,[[77,null,null,[1]],[78,null,null,[1]],[85,null,null,[1]],[76,null,null,[1]]]]],[4,null,6,null,null,null,null,["21061508"]]],[500,[[21060697],[21060698,[[87,null,null,[1]]]]],[2,[[4,null,6,null,null,null,null,["21061508"]],[4,null,8,null,null,null,null,["Uint8Array"]],[4,null,11]]]],[100,[[21061497],[21061498,[[86,null,null,[1]]]]],[2,[[4,null,6,null,null,null,null,["21061508"]],[4,null,9,null,null,null,null,["requestAnimationFrame"]]]]],[100,[[21061545],[21061546,[[79,null,null,[1]]]]],[2,[[4,null,6,null,null,null,null,["21061508"]],[4,null,8,null,null,null,null,["google_ltobserver"]]]]],[50,[[21061863,[[null,1,null,[null,4096],[[[4,null,14],[null,8192]]]]]],[21061864,[[null,1,null,[null,12288]]]],[21061865,[[null,1,null,[null,15360]]]]]],[50,[[21061999],[21062000,[[81,null,null,[1]]]]],[2,[[4,null,6,null,null,null,null,["21061508"]],[4,null,10]]]],[null,[[21062185],[21062186,[[24,null,null,[1]]]]]],[1,[[21062330],[21062331,[[null,8,null,[null,800]]]],[21062332,[[null,8,null,[null,10000]]]]],null,3],[50,[[21062414],[21062415,[[64,null,null,[1]]]]]],[50,[[21062420],[21062421,[[42,null,null,[1]]]]]],[50,[[21062452],[21062453,[[43,null,null,[1]]]]]],[50,[[21062724],[21062725,[[67,null,null,[1]]]]]],[10,[[21062751],[21062752,[[null,15,null,[null,1]]]],[21062753,[[null,15,null,[null,2]]]]]],[10,[[21062796],[21062797,null,[4,null,8,null,null,null,null,["Map"]]]]],[50,[[21062818],[21062819,[[93,null,null,[1]]]]]],[50,[[21062832],[21062833,[[89,null,null,[1]]]]]],[50,[[21062886],[21062887,[[91,null,null,[1]]]]]],[10,[[21062888],[21062889,[[101,null,null,[1]]]]]],[5,[[21062899],[21062900,[[98,null,null,[1]]]],[21062901,[[98,null,null,[1]]]]]],[5,[[21062916,[[98,null,null,[1]]]],[21062917,[[98,null,null,[1]]]]]],[1,[[21062970],[21062971,[[109,null,null,[1]]]]]],[50,[[21063015],[21063016,[[97,null,null,[1]]]]]],[5,[[21063046],[21063047,[[142,null,null,[1]]]],[21063048,[[142,null,null,[1]]]]],[2,[[4,null,7],[4,null,8,null,null,null,null,["TextDecoder"]],[4,null,10]]],9],[null,[[21063065],[21063066,[[116,null,null,[1]]]]]],[null,[[21063094],[21063095,[[142,null,null,[1]]]],[21063096,[[142,null,null,[1]]]]],[2,[[4,null,7],[4,null,8,null,null,null,null,["TextDecoder"]],[4,null,10]]],9],[1,[[21063145],[21063146,[[112,null,null,[1]]]]]],[1,[[21063147],[21063148,[[99,null,null,[1]]]]]],[1000,[[21063165,null,[3,[[6,null,null,1,null,0],[6,null,null,1,null,5]]]],[21063166,[[114,null,null,[1]]],[3,[[6,null,null,1,null,1],[6,null,null,1,null,6]]]]],[4,null,3]],[10,[[21063167],[21063168,[[102,null,null,[1]]]]]],[50,[[21063202],[21063203,[[123,null,null,[1]]]]]],[10,[[21063204],[21063205,[[47,null,null,[1]]]]]],[1000,[[21063316,null,[3,[[6,null,null,1,null,2],[6,null,null,1,null,7]]]],[21063317,[[114,null,null,[1]]],[3,[[6,null,null,1,null,3],[6,null,null,1,null,8]]]]],[4,null,3]],[5,[[21063340],[21063341,[[129,null,null,[1]],[65,null,null,[1]]]],[21063342,[[129,null,null,[1]],[65,null,null,[1]],[71,null,null,[1]]]]]],[50,[[21063387],[21063388,[[130,null,null,[1]]]]]],[1,[[21063633],[21063634,[[143,null,null,[1]]]]],[2,[[4,null,10]]]],[50,[[21063635],[21063636,[[104,null,null,[1]]]]]],[10,[[21063637],[21063638,[[141,null,null,[1]]]]]],[1,[[21063669],[21063670],[21063671,[[142,null,null,[1]]]]],[4,null,8,null,null,null,null,["TextDecoder"]],9],[1,[[21063778],[21063779,[[132,null,null,[1]],[110,null,null,[1]]]]],null,11],[1,[[21063792],[21063793,[[148,null,null,[1]]]]]],[50,[[21063817],[21063818,[[149,null,null,[1]]]]]],[10,[[21063910],[21063911]]],[1,[[21063964],[21063965,[[156,null,null,[1]]]],[21063966,[[157,null,null,[1]]]],[21063967,[[156,null,null,[1]],[157,null,null,[1]]]]]],[50,[[21064055],[21064056,[[110,null,null,[1]]]]],null,11],[null,[[21064078],[21064079,[[null,null,null,[null,null,null,["v","1-0-35"]],null,1]]],[21064080,[[null,null,2,[null,null,"1-0-35"]]]]]],[10,[[21064100],[21064101,[[163,null,null,[1]]]]]],[10,[[21064165],[21064166]]],[50,[[21064169],[21064170,[[168,null,null,[1]]]]]],[1000,[[21064180,null,[4,null,6,null,null,null,null,["21064177"]]],[21064181,null,[4,null,6,null,null,null,null,["21064178"]]],[21064182,null,[4,null,6,null,null,null,null,["21064179"]]]],[2,[[4,null,16],[4,null,9,null,null,null,null,["Promise"]],[4,null,9,null,null,null,null,["IntersectionObserver"]]]]],[1,[[21064194],[21064195,[[165,null,null,[1]]]]]],[50,[[21064225],[21064226,[[null,13,null,[]]]]]],[1,[[21064227],[21064228,[[159,null,null,[1]],[139,null,null,[1]]]]]],[1000,[[22316437,null,[2,[[8,null,null,1,null,-1],[7,null,null,1,null,5]]]],[22316438,null,[2,[[8,null,null,1,null,4],[7,null,null,1,null,10]]]]],[4,null,3]],[100,[[22325465],[22325466,[[80,null,null,[1]]]]],[4,null,6,null,null,null,null,["21060611"]]],[1,[[108809132],[108809133,[[45,null,null,[1]]]]]],[10,[[370204026],[370204027],[370204053]]],[10,[[370204075],[370204076,[[175,null,null,[1]]]]]]]],[4,[[null,[[21063411],[21063412],[21063413]]],[null,[[21063421],[21063422],[21063423,[[74,null,null,[1]]]]]],[null,[[21063599],[21063600,[[105,null,null,[1]]]],[21064038],[21064039,[[105,null,null,[1]]]]]],[null,[[21063829,[[150,null,null,[1]]]]]],[null,[[21063831],[21063832],[21063833,[[null,19,null,[null,30]]]],[21063834,[[null,19,null,[null,30]],[150,null,null,[1]]]]]],[null,[[21063927],[21063928,[[null,16,null,[null,500]]]],[21063929,[[null,16,null,[null,500]]]],[21063930,[[null,16,null,[null,750]]]],[21063931,[[null,16,null,[null,1000]]]],[21063932,[[null,17,null,[null,50]]]],[21063933,[[null,17,null,[null,100]]]],[21063934,[[null,17,null,[null,150]]]],[21063935,[[null,17,null,[null,200]]]],[21063936,[[null,18,null,[null,1]]]],[21063937,[[null,18,null,[null,250]]]],[21063938,[[null,18,null,[null,500]]]],[21063939,[[null,18,null,[null,750]]]],[21063940,[[null,18,null,[null,1000]]]]]],[null,[[21063941],[21063942,[[null,16,null,[null,250]]]],[21063943,[[null,16,null,[null,500]]]],[21063944,[[null,16,null,[null,750]]]],[21063945,[[null,16,null,[null,1000]]]],[21063946,[[null,17,null,[null,50]]]],[21063947,[[null,17,null,[null,100]]]],[21063948,[[null,17,null,[null,150]]]],[21063949,[[null,17,null,[null,200]]]],[21063950,[[null,18,null,[null,250]]]],[21063951,[[null,18,null,[null,500]]]],[21063952,[[null,18,null,[null,750]]]],[21063953,[[null,18,null,[null,1000]]]]]],[null,[[21063962],[21063963,[[null,18,null,[null,1]]]]]],[null,[[21063987],[21063988]]],[null,[[21064027],[21064028,[[105,null,null,[1]]]]]],[null,[[21064059,[[null,22,null,[null,30]]]]]],[null,[[21064154],[21064155,[[null,null,null,[null,null,null,["1288355901"]],null,5],[null,null,null,[null,null,null,["AjFHi2xI34QG9mkTo+LAkUveOiwZ5PA431Mg5xMZgzDG9ILu992s838MxmWTxC5VXcTZ8BLhuaCHUa03Ru8fUwQAAABneyJvcmlnaW4iOiJodHRwczovL3d3dy5vdWVzdC1mcmFuY2UuZnI6NDQzIiwiZmVhdHVyZSI6IkV4cGVyaW1lbnRhbElzSW5wdXRQZW5kaW5nIiwiZXhwaXJ5IjoxNTY1ODAyNzI4fQ=="]],null,6],[169,null,null,[1]]]]],[4,null,17,null,null,null,null,["1288355901"]]],[null,[[676982417]]],[null,[[676982603],[676982604]]],[null,[[676982661],[676982662],[676982663]]],[null,[[676982666],[676982667],[676982669],[676982670]]],[null,[[676982672],[676982673],[676982674],[676982675],[676982676],[676982677]]],[null,[[676982681]]]]],[5,[[10,[[21061507],[21061508,[[83,null,null,[1]],[84,null,null,[1]]]]]],[1000,[[21062785,[[23,null,null,[]]],[7,null,null,5,null,50]],[21062786,[[23,null,null,[1]]],[8,null,null,5,null,949]]],[2,[[12,null,null,null,2,null,"today\\.line\\.me/.+/article"],[4,null,8,null,null,null,null,["_gmptnl"]]]],7],[1000,[[21062812,[[23,null,null,[1]]],[2,[[8,null,null,5,null,49],[7,null,null,5,null,950]]]]],[2,[[12,null,null,null,2,null,"today\\.line\\.me/.+/article"],[4,null,8,null,null,null,null,["_gmptnl"]]]],7],[50,[[21063232,null,[6,null,null,null,4,null,"slow-2g",["navigator.connection.effectiveType"]]]],[5,null,null,null,4,null,null,["navigator.connection.effectiveType"]]],[50,[[21063233,null,[6,null,null,null,4,null,"2g",["navigator.connection.effectiveType"]]]],[5,null,null,null,4,null,null,["navigator.connection.effectiveType"]]],[50,[[21063234,null,[6,null,null,null,4,null,"3g",["navigator.connection.effectiveType"]]]],[5,null,null,null,4,null,null,["navigator.connection.effectiveType"]]],[50,[[21063235,null,[6,null,null,null,4,null,"4g",["navigator.connection.effectiveType"]]]],[5,null,null,null,4,null,null,["navigator.connection.effectiveType"]]],[50,[[21063247]],[1,[[5,null,null,null,4,null,null,["navigator.connection.effectiveType"]]]]],[50,[[21063248,null,[1,[[12,null,null,null,4,null,"^(slow-2g|2g|3g|4g)$",["navigator.connection.effectiveType"]]]]]],[5,null,null,null,4,null,null,["navigator.connection.effectiveType"]]],[1000,[[21063916,[[23,null,null,[]]],[7,null,null,5,null,500]],[21063917,[[23,null,null,[1]]],[8,null,null,5,null,499]]],[2,[[12,null,null,null,2,null,"today\\.line\\.me/.+/main"],[4,null,8,null,null,null,null,["webkit.messageHandlers._gmptnl"]]]],7],[10,[[21064029,null,[4,null,8,null,null,null,null,["webkit.messageHandlers._gmptnl"]]]]],[10,[[21064030,null,[4,null,8,null,null,null,null,["_gmptnl"]]]]],[1,[[21064177,[[null,null,5,[null,null,"21064177"]],[null,null,6,[null,null,"21064177"]]]],[21064178,[[null,null,4,[null,null,"/gpt/pubads_impl_legacy_"]],[null,null,5,[null,null,"21064178"]],[null,null,6,[null,null,"21064178"]]]],[21064179,[[null,null,4,[null,null,"/gpt/pubads_impl_legacy_"],[[[2,[[4,null,16],[4,null,9,null,null,null,null,["Promise"]],[4,null,9,null,null,null,null,["IntersectionObserver"]]]],[null,null,"/gpt/pubads_impl_modern_"]]]],[null,null,5,[null,null,"21064179"]],[null,null,6,[null,null,"21064179"]]]]],[3,[[6,null,null,4,null,1],[6,null,null,4,null,0]]],1],[1000,[[21064196,[[null,7,null,[null,1]],[null,null,5,[null,null,"21064196"]],[60,null,null,[1]],[null,null,6,[null,null,"21064196"]]],[6,null,null,4,null,2]],[21064197,[[null,7,null,[null,1]],[60,null,null,[1]]],[6,null,null,4,null,3]]],[4,null,3],1],[1000,[[21064219,[[null,7,null,[null,1]],[null,null,5,[null,null,"21064219"]],[60,null,null,[1]],[null,null,6,[null,null,"21064219"]]],[6,null,null,4,null,4]],[21064220,[[null,7,null,[null,1]],[60,null,null,[1]]],[6,null,null,4,null,5]]],[4,null,3],1]]],[6,[[null,[[21062379,[[23,null,null,[1]]]]]],[10,[[21063049],[21063050],[21063051]],[3,[[4,null,6,null,null,null,null,["21062415"]],[4,null,6,null,null,null,null,["21062414"]]]]],[50,[[21064102],[21064103,[[159,null,null,[1]]]]],[2,[[4,null,12]]]]]],[9,[[1000,[[21061726]],[4,null,13,null,null,null,null,["PnHSZjekOp","jvnwkvnp"]]]]]]])