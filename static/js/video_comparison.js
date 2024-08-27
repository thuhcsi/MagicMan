// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoIds) {
    function drawLoop() {
        var colStart = (vidWidth * position).clamp(0.0, vidWidth);
        var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);

        for (var i = 0; i < mergeContexts.length; i++) {
            mergeContexts[i].drawImage(vids[i], 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
            mergeContexts[i].drawImage(vids[i], colStart + vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
        }
        requestAnimationFrame(drawLoop);


        var arrowLength = 0.09 * vidWidth;
        var arrowheadWidth = 0.025 * vidWidth;
        var arrowheadLength = 0.04 * vidWidth;
        var arrowPosY = vidHeight / 2;
        var arrowWidth = 0.007 * vidWidth;
        var currX = vidWidth * position;

        for (var mergeContext of [mergeContexts[0]]) {
            // Draw circle
            // mergeContext.arc(currX, arrowPosY, arrowLength * 0.7, 0, Math.PI * 2, false);
            // mergeContext.arc(currX, arrowPosY, vidWidth*0.04, 0, Math.PI * 2, false);
            mergeContext.arc(currX, arrowPosY, vidWidth*0.045, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#ffd793af";
            // mergeContext.fillStyle = "#8ccce7ca";
            // mergeContext.fillStyle = "#b6e59695";
            mergeContext.fill()
            //mergeContext.strokeStyle = "#444444";
            //mergeContext.stroke()

            // Draw border
            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth * position, 0);
            mergeContext.lineTo(vidWidth * position, vidHeight);
            mergeContext.closePath()
            // mergeContext.strokeStyle = "#444444";
            mergeContext.strokeStyle = "#21212195";
            mergeContext.lineWidth = vidWidth / 200;
            mergeContext.stroke();

            // Draw arrow
            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth / 2);

            // Move right until meeting arrow head
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowWidth / 2);

            // Draw right arrow head
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2, arrowPosY);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowWidth / 2);

            // Go back to the left until meeting left arrow head
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowWidth / 2);

            // Draw left arrow head
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2, arrowPosY);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY);

            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowWidth / 2);
            mergeContext.lineTo(currX, arrowPosY - arrowWidth / 2);

            mergeContext.closePath();

            mergeContext.fillStyle = "#565656";
            mergeContext.fill();
        }

    }
    var videoMerges = [];
    var mergeContexts = [];
    var vids = [];
    for (const videoId of videoIds) {
        var element = document.getElementById(videoId + "_merge");
        videoMerges.push(document.getElementById(videoId + "_merge"));
        vids.push(document.getElementById(videoId));
        mergeContexts.push(element.getContext("2d"));
    }
    var position = 0.5;
    var vidWidth = vids[0].videoWidth / 2;
    var vidHeight = vids[0].videoHeight;

    for (const vid of vids){
        if (vid.readyState > 3) {
            vid.play();
        }
    }

    for (const videoMerge of [videoMerges[0]]) {
        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
            position = position % 1
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - bcr.x) / bcr.width);
            position = position % 1
        }

        videoMerge.addEventListener("mousemove", trackLocation, false);
        videoMerge.addEventListener("touchstart", trackLocationTouch, false);
        videoMerge.addEventListener("touchmove", trackLocationTouch, false);
    
    }
    requestAnimationFrame(drawLoop);

}

Number.prototype.clamp = function (min, max) {
    return Math.min(Math.max(this, min), max);
};

function preprocess(element) {
    var cv = document.getElementById(element.id + "_merge");
    cv.width = element.videoWidth / 2;
    cv.height = element.videoHeight;

    // console.log("video height:" + element.videoHeight)
    // console.log("video width:" + element.videoWidth)

    element.play();
    element.style.height = "0px";  // Hide video without stopping it
    return element
}


function resizeAndPlay(element, videoIds = null) {
    element = preprocess(element);
    all_videoIds = [element.id]
    if (videoIds) {
        for (const videoId of videoIds) {
            new_element = document.getElementById(videoId);
            new_element = preprocess(new_element);
            all_videoIds.push(new_element.id);
        }
    }
    // console.log(all_videoIds)
    playVids(all_videoIds);
}

