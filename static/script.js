// Below we've defined global variables of various HTML elements
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

// If statement at beginning ensures that Spell-correcting features not shown if there's no output text shown
// Will be checked once (as this is all that is required) when page is rendered
if (outputText.innerHTML === "") {
    spellingSwitch.style.display = "none";
    spellingLabel.style.display = "none";
}


uploadButton.onchange = () => {
    /*
        Function called when upload button pressed. It opens file directory on computer and when file is
        uploaded it displays it and the filename below.
     */
    let reader = new FileReader();
    reader.readAsDataURL(uploadButton.files[0]);
    reader.onload = () => {
        // Sets image element's source to the uploaded file so it displays
        chosenImage.setAttribute("src", reader.result);
        //segmentedImage.style.display = "none";
        segmentedImageFigure.style.display = "none";
    }
    // Allows file name to appear below as label
    fileName.textContent = uploadButton.files[0].name;


}


function loading(){
    /*
        Function called after convert text button is pressed. Function hides certain elements which shouldn't
        be on the screen and also runs loading animation.
     */
    // Hides elements - useful for loading animation to be centred in screen and not displaced by output text
    outputText.style.display = "none";
    spellingOutput.style.display = "none";
    spellingSwitch.style.display = "none";
    spellingLabel.style.display = "none";

    // We use a for loop to apply animation to each dot element and add different delays to create wave-live animation
    let dot_count = dots.length;
    for (let i = 0; i < dot_count; i++){
        dots[i].style.animation = "loading 1.8s ease-in-out infinite";
        dots[i].style.animationDelay = String(0.2 * i).concat("s");
    }

    waitMessage.innerHTML = "Conversion will finish in around 10 seconds";

}

function spellCheckClicked() {
    /*
        Function called when toggle button is toggled. Function displays either the spell-corrected or the direct
        predictions from the model depending on what state the toggle switch is.
     */
    // Gets the switch input element to check its state
    let checkBox = document.getElementById("spellcheck-input");

    if (checkBox.checked == true){
        // If the toggle switch is on - turn label to "Spell-checker: on" in green and show spell-corrected text
        spellingLabel.style.color = "green";
        spellingLabel.innerHTML = "Spell-checker: on";
        outputText.style.display = "none";
        spellingOutput.style.display = "block";
    }
    else{
        // If the toggle switch is off - turn label to "Spell-checker: off" in red and show predicted text straight from model
        spellingLabel.style.color = "red";
        spellingLabel.innerHTML = "Spell-checker: off";
        spellingOutput.style.display = "none";
        outputText.style.display = "block";
    }

}