let uploadButton = document.getElementById("uploadImage");
let chosenImage = document.getElementById("chosen-image");
let fileName = document.getElementById("file-name");

let segmentedImage = document.getElementById("segmented-image");
let convertButton = document.getElementById("convertButton");

let dots = document.getElementsByTagName("span");
let waitMessage = document.getElementById("waitMessage");

uploadButton.onchange = () => {
    let reader = new FileReader();
    reader.readAsDataURL(uploadButton.files[0]);
    reader.onload = () => {
        chosenImage.setAttribute("src", reader.result);
        segmentedImage.style.display = "none";
    }
    fileName.textContent = uploadButton.files[0].name


}


function loading(){
    let dot_count = dots.length;
    for (let i = 0; i < dot_count; i++){
        dots[i].style.animation = "loading 1.8s ease-in-out infinite";
        dots[i].style.animationDelay = String(0.2 * i).concat("s");
    }

    waitMessage.innerHTML = "Conversion will finish in around 5 seconds"

}