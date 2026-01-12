const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const log = document.getElementById("log");

let filename = null;
let mode = "video";
let counter = 1;

function start() {
    mode = document.getElementById("mode").value;
    log.innerHTML = "";
    counter = 1;

    if (mode === "video") {
        uploadVideo();
    } else {
        startCamera();
    }
}

function uploadVideo() {
    const file = document.getElementById("videoFile").files[0];
    const fd = new FormData();
    fd.append("video", file);

    fetch("/upload", { method: "POST", body: fd })
        .then(r => r.json())
        .then(d => {
            filename = d.filename;
            video.src = `/video/${filename}`;
            video.playbackRate = 0.5;
            video.play();
        });
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.play();
        });
}

video.addEventListener("loadedmetadata", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
});

setInterval(() => {
    if (video.paused || video.ended) return;

    fetch("/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            source: mode,
            filename: filename,
            time: video.currentTime
        })
    })
    .then(r => r.json())
    .then(detections => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            ctx.strokeStyle = "lime";
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.fillStyle = "lime";
            ctx.fillText(det.plate, x1, y1 - 5);

            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${counter++}</td>
                <td>${det.plate}</td>
                <td>${det.confidence}</td>
                <td>${video.currentTime.toFixed(1)}</td>
            `;
            log.appendChild(row);
        });
    });
}, 500);
