// "use strict";
//
// let map;
// $(document).ready(function () {
//     let latitude = 40.91;
//     let longitude = -96.63;
//
//     if (map != undefined) {
//         map.off();
//         map.remove();
//     }
//     if (navigator.geolocation) {
//         navigator.geolocation.getCurrentPosition(function (position) {
//             // latitude = position.coords.latitude;
//             longitude = position.coords.longitude;
//
//             map = setupMap({latitude, longitude}, 14);
//             let circle = L.circle([latitude, longitude], {
//                 color: 'red',
//                 fillColor: '#ff337d',
//                 fillOpacity: 0.3,
//                 radius: 500,
//                 opacity: 0.3
//             }).addTo(map);
//         });
//     } else {
//         map = setupMap({latitude, longitude}, 4);
//     }
// });
//
//
// function setupMap(location, zoomLevel) {
//     map = L.map('map');
//
//     map.on('load', function () {
//         L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
//         setInterval(function () {
//             map.invalidateSize();
//         }, 1000);
//     });
//     map.setView([location.latitude, location.longitude], zoomLevel);
//
//     // L.esri.basemapLayer('Imagery').addTo(map);
//     // L.esri.basemapLayer('ImageryLabels').addTo(map);
//
//
//     let searchControl = L.esri.Geocoding.geosearch().addTo(map);
//
//     let results = L.layerGroup().addTo(map);
//
//     let latitude = $('#latitude');
//     let longitude = $('#longitude');
//
//     searchControl.on('results', function (data) {
//         results.clearLayers();
//         for (let i = data.results.length - 1; i >= 0; i--) {
//             let latlng = data.results[i].latlng;
//
//             results.addLayer(L.marker(latlng));
//             latitude.val(latlng.lat);
//             longitude.val(latlng.lng);
//
//             console.debug('Latitude: ' + latlng.lat);
//             console.debug('Longitude: ' + latlng.lng);
//
//         }
//     });
//
//     map.on('click', function (e) {
//         results.clearLayers();
//
//         let latlng = e.latlng;
//
//         results.addLayer(L.marker(latlng));
//
//         latitude.val(latlng.lat);
//         longitude.val(latlng.lng);
//
//         console.debug('Latitude: ' + latlng.lat);
//         console.debug('Longitude: ' + latlng.lng);
//
//     });
//
//     return map;
// }
//
// function setupPostEditMap(location, zoomLevel) {
//     map = L.map('map');
//
//     map.on('load', function () {
//         L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
//         setInterval(function () {
//             map.invalidateSize();
//         }, 1000);
//     });
//     map.setView([location.latitude, location.longitude], zoomLevel);
//
//     // L.esri.basemapLayer('Imagery').addTo(map);
//     // L.esri.basemapLayer('ImageryLabels').addTo(map);
//
//
//     let searchControl = L.esri.Geocoding.geosearch().addTo(map);
//
//     let results = L.layerGroup().addTo(map);
//     results.addLayer(L.marker({lat:latitude, lng: longitude}))
//
//     let latitude = $('#latitude');
//     let longitude = $('#longitude');
//
//     searchControl.on('results', function (data) {
//         results.clearLayers();
//         for (let i = data.results.length - 1; i >= 0; i--) {
//             let latlng = data.results[i].latlng;
//
//             results.addLayer(L.marker(latlng));
//             latitude.val(latlng.lat);
//             longitude.val(latlng.lng);
//
//             console.debug('Latitude: ' + latlng.lat);
//             console.debug('Longitude: ' + latlng.lng);
//
//         }
//     });
//
//     map.on('click', function (e) {
//         results.clearLayers();
//
//         let latlng = e.latlng;
//
//         results.addLayer(L.marker(latlng));
//
//         latitude.val(latlng.lat);
//         longitude.val(latlng.lng);
//
//         console.debug('Latitude: ' + latlng.lat);
//         console.debug('Longitude: ' + latlng.lng);
//
//     });
//
//     return map;
// }