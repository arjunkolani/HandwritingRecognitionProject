let uploadButton = document.getElementById("uploadImage");
let chosenImage = document.getElementById("chosen-image");
let fileName = document.getElementById("file-name");

//let segmentedImage = document.getElementById("segmented-image");
let segmentedImageFigure = document.getElementById("segmented-image-figure");
let convertButton = document.getElementById("convertButton");

let dots = document.getElementsByClassName("loadingDot");
let waitMessage = document.getElementById("waitMessage");

let outputText = document.getElementById("output-text");

let spellingSwitch = document.getElementById("spellcheck-switch");
let spellingLabel = document.getElementById("spellcheck-label");
let spellingOutput = document.getElementById("spellcheck-output");

if (outputText.innerHTML === "") {
    spellingSwitch.style.display = "none";
    spellingLabel.style.display = "none";
}


uploadButton.onchange = () => {
    let reader = new FileReader();
    reader.readAsDataURL(uploadButton.files[0]);
    reader.onload = () => {
        chosenImage.setAttribute("src", reader.result);
        //segmentedImage.style.display = "none";
        segmentedImageFigure.style.display = "none";
    }
    fileName.textContent = uploadButton.files[0].name;


}


function loading(){
    outputText.style.display = "none";
    spellingOutput.style.display = "none";
    spellingSwitch.style.display = "none";
    spellingLabel.style.display = "none";

    //convertButton.disabled = true;
    let dot_count = dots.length;
    for (let i = 0; i < dot_count; i++){
        dots[i].style.animation = "loading 1.8s ease-in-out infinite";
        dots[i].style.animationDelay = String(0.2 * i).concat("s");
    }

    waitMessage.innerHTML = "Conversion will finish in around 10 seconds";

}

function spellCheckClicked() {
    let checkBox = document.getElementById("spellcheck-input");

    if (checkBox.checked == true){
        spellingLabel.style.color = "green";
        spellingLabel.innerHTML = "Spell-checker: on";
        outputText.style.display = "none";
        spellingOutput.style.display = "block";
    }
    else{
        spellingLabel.style.color = "red";
        spellingLabel.innerHTML = "Spell-checker: off";
        spellingOutput.style.display = "none";
        outputText.style.display = "block";
    }

}