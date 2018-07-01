const CACHE_NAME = 'kpspemu-v7';
const urlsToCache = [
    '/',
    '/index.html',
    '/manifest.json',
    '/service-worker.js',
    '/apple-touch-icon.png',
    '/lucida_console32.fnt',
    '/lucida_console32.png',
    '/icon-48.png',
    '/icon-96.png',
    '/icon-144.png',
    '/icon-196.png',
    '/icon-384.png',
    '/icon-512.png',
    '/require.min.js',
    '/dynarek-js.js',
    '/kds-js.js',
    '/klock-js.js',
    '/klogger-js.js',
    '/kmem-js.js',
    '/korag-js.js',
    '/korau-js.js',
    '/korau-atrac3plus-js.js',
    '/korge-js.js',
    '/korim-js.js',
    '/korinject-js.js',
    '/korio-js.js',
    '/korma-js.js',
    '/korui-js.js',
    '/kotlin.js',
    '/kpspemu.js',
    '/krypto-js.js',
    '/minifire.elf',
    '/kzlib-js.js'
];
console.log('loading service worker');

self.addEventListener('install', function (event) {
    // Perform install steps
    console.log('installing service worker');

    async

    function load() {
        try {
            const cache = await
            caches.open(CACHE_NAME);
            console.log('Opened cache ', CACHE_NAME);
            const tasks = [];
            for (const url of urlsToCache) {
                console.log(` - ${url}`);
                tasks.push({url: url, promise: cache.add(new Request(url))});
            }
            for (const task of tasks) {
                try {
                    await
                    task.promise;
                    console.log(' - cache added: ', task.url);
                } catch (e) {
                    console.error(' - cache error: ', task.url);
                }
            }
        } catch (e) {
            console.error('error installing', e);
            throw e
        }
    }

    event.waitUntil(load());
});

self.addEventListener('fetch', function (event) {
    console.log('Fetching...', CACHE_NAME, event.request);
    async

    function myfetch() {
        try {
            const response = await
            caches.match(event.request);
            // Cache hit - return response
            if (response) {
                return response;
            }
            return await
            fetch(event.request);
            //const cache = await caches.open(CACHE_NAME);
            //return await cache.add(event.request);
        } catch (e) {
            console.error('error fetching', e);
            throw e
        }
    }

    event.respondWith(myfetch());
});

self.addEventListener('activate', function (event) {
    console.log('Service Worker: activated');
    // Perform some task
});

self.addEventListener('message', function (event) {
    console.log("SW Received Message: " + event.data);
    switch (event.data) {
        case 'refresh':
            console.log('deleted cache ', CACHE_NAME);
            caches.delete(CACHE_NAME);
            break;
    }
});
