window.addEventListener('load', init);

let fileData;
let fileDetecter;
let textView;
let outTextView;
let calcButton;
let testButton;

let inputData;
let targetData = new Array();
let dataIndex = 0;

const bias = 1;
const leraningRate = 0.01;
const MULIT_LAYER_SIZE = 3;
const INPUT_LAYER_SIZE = 4;
const HIDDEN_LAYER_SIZE = 3;
const OUTPUT_LAYER_SIZE = 3;

let hiddenLayerWeight;
let outputLayerWeight;

let hiddenLayerOutput;
let outputLayerOutput = new Array();

let outputLayerDelta = new Array();
let hiddenLayerDelta = new Array();

const LOOP_VALUE = 500000;
let trainningValue = new Array();
let continueNumber = 0;
let testTry = 0;

function init() {
    fileDetecter = document.querySelector('input#file-input');
    textView = document.getElementById('text_input');
    outTextView = document.getElementById('output_text');

    calcButton = document.getElementById('calcButton');
    calcButton.addEventListener('click', calcButtonHandler);
    testButton = document.getElementById('testButton');
    testButton.addEventListener('click', testButtonHendler);

    hiddenLayerWeight = initWeight(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE);
    outputLayerWeight = initWeight(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);

    fileDetecter.addEventListener('change', function (event) {
        let file = event.target.files[0];
        if (!file) {
            alert('not File!');
            return;
        }

        let reader = new FileReader();
        reader.onload = function (e) {
            //    fileData = e.target.result;
            textView.innerText = e.target.result;
            inputData = [];
            inputData = textLoadHandler(INPUT_LAYER_SIZE);
        }
        reader.readAsText(file);
    });

}//init


/**
 * 파일을 읽은 후 처리할 수 있는 데이터로 변환하는 선작업 함수.
 * @param {int} inputSize input 개수
 */
function textLoadHandler(inputSize) {

    let inputData_text = textView.innerText.split(' ');
    let dataLength = inputData_text.length / inputSize;
    let tempInput_data = new Array();
    for (let i = 0; i < dataLength; i++) {
        targetData[i] = (i < 25 ? [1, 0, 0] : i < 50 ? [0, 1, 0] : [0, 0, 1]);
        trainningValue[i] = 99;
    }

    let data = new Array();
    for (let i = 0; i < inputSize; i++) {
        data[i] = Number(inputData_text[i]);
    }
    tempInput_data.push(data)

    for (let i = 1; i < dataLength; i++) {
        let tempArray = new Array();
        for (let j = 0, offset = i * inputSize; j < inputSize; j++ , offset++) {
            tempArray[j] = (offset % 4 == 0 ?
                Number(inputData_text[offset].split('\n')[1]) :
                Number(inputData_text[offset]));
        }
        tempInput_data.push(tempArray);
    }

    return tempInput_data;
}

/**
 * 가중치 초기화
 * @param {int} outLayerLength 출력 레이어 개수
 * @param {int} inputLayerLength 입력 레이어 개수
 */
function initWeight(outLayerLength, inputLayerLength) {

    let neural = new Array();

    for (let i = 0; i < outLayerLength; i++) {
        let weights = new Array();
        for (let j = 0; j < inputLayerLength; j++) {
            weights[j] = Math.random() - 0.5;
        }
        neural[i] = weights;
    }
    return neural;
}

function testButtonHendler() {
    printAlltestOutput();
}

function calcButtonHandler() {
    printAllOutput();
}

/**
 * 계산 총괄
 * @param {bool} doUpdate 학습중 true
 */
function calculateOutput(doUpdate) {
    hiddenLayerOutput = getOutput(hiddenLayerWeight, inputData[dataIndex]);

    let outputLayerOutputTempArray = getOutput(outputLayerWeight, hiddenLayerOutput);
    outputLayerOutput[dataIndex] = outputLayerOutputTempArray;

    if (doUpdate)
        updateWeight(outputLayerOutputTempArray);

    dataIndex = (dataIndex + 1) % inputData.length;
}


function getOutput(weights, inputs) {
    let outputArray = new Array();

    for (let i = 0; i < weights.length; i++) {
        outputArray[i] = bias;
        for (let j = 0; j < inputs.length; j++) {
            outputArray[i] += inputs[j] * weights[i][j];
        }
        outputArray[i] = sigmoide(outputArray[i]);
    }

    return outputArray;
}

function printAllOutput() {
    outTextView.innerText = "";
    dataIndex = 0;
    
    for (let i = 0; i < inputData.length * LOOP_VALUE; i++) {
        /*
        let offset = (dataIndex < 25 ? 0 : dataIndex < 50 ? 1 : 2);

        testTry++;
        if (trainningValue[dataIndex] == offset) {
            dataIndex = (dataIndex + 1) % inputData.length;
            continueNumber ++;
            continue;
        }
        */
        calculateOutput(true);
    }
    outTextView.innerText += 'try :' + (testTry / inputData.length) + '\n';
    for (let i = 0; i < inputData.length; i++) {
        printOutput(i);
    }
}

function printAlltestOutput() {
    outTextView.innerText = "";
    dataIndex = 0;

    for (let i = 0; i < inputData.length; i++) {
        calculateOutput(false);
        printOutput(i);
    }
}

function printOutput(dataNumber) {
    trainningValue[dataNumber] = outputLayerOutput[dataNumber].indexOf(Math.max(...outputLayerOutput[dataNumber]));
    outTextView.innerText += 'dataSet :' + dataNumber + "    index: " + trainningValue[dataNumber];
    outTextView.innerText += '\n';
}

function updateWeight(outputLayerValue) {

    for (let i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        outputLayerDelta[i] = (targetData[dataIndex][i] - outputLayerValue[i]) * outputLayerValue[i] * (1 - outputLayerValue[i]);
        for (let j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            outputLayerWeight[i][j] += leraningRate * outputLayerDelta[i] * hiddenLayerOutput[j];
        }
    }

    for (let i = 0; i < OUTPUT_LAYER_SIZE; i++) {
        hiddenLayerDelta[i] = 0;
        for (let j = 0; j < OUTPUT_LAYER_SIZE; j++) {
            hiddenLayerDelta[i] += outputLayerWeight[j][i] * outputLayerDelta[i];
        }
    }

    for (let i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        for (let j = 0; j < INPUT_LAYER_SIZE; j++) {
            hiddenLayerWeight[i][j] += leraningRate * hiddenLayerDelta[i] * hiddenLayerOutput[i]
                * (1 - hiddenLayerOutput[i]) * inputData[dataIndex][j];
        }
    }
}

function sigmoide(input) {
    return 1 / (1 + Math.pow(Math.E, -input));
}
