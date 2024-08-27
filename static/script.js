let uploadButton = document.getElementById("uploadImage");
let chosenImage = document.getElementById("chosen-image");
let fileName = document.getElementById("file-name");

let segmentedImage = document.getElementById("segmented-image")
let convertButton = document.getElementById("convertButton");


uploadButton.onchange = () => {
    let reader = new FileReader();
    reader.readAsDataURL(uploadButton.files[0]);
    reader.onload = () => {
        chosenImage.setAttribute("src", reader.result);
        segmentedImage.style.display = "none";
    }
    fileName.textContent = uploadButton.files[0].name;
}

convertButton.onchange = () => {
    segmentedImage.style.display = "block";
}