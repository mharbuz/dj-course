import { addMatrices, multiplyMatrices, transpose, assertMatricesDimensionMatch, assertMatricesCompatible } from "./matrix-operations";
import { fromJSONFile, jsonFilePath, randomizeMatrix, randomizeVector } from "./utils";
import { vectorSum, dotProduct } from "./vector-operations";
import { Matrix, Vector } from "./types";
import { displayVector, displayMatrix } from "./display";

// HINT: (w zaleności od wybranego kierunku implementacji) może być mnożenie macierzy przez wektory - tę operację będzie trzeba zaimplementować 😉 
// ale nie jest to konieczne 😎

// HINT: w mnożeniu macierzy kolejność ma znaczenie - bo w zależności od kolejności albo wymiary obydwu składników pasują do siebie albo nie.

// HINT: wstań od komputera i przemyśl problem. Serio. Zastanów się, ile linijek wystarczy aby podać rozwiązanie :)
// (traktując "linijkę" jako pojedynczą operację na tensorach) 😎

// PROŚBA: jeśli znasz rozwiązanie, to nie spamuj discorda - a przynajmniej nie od razu. Pozwól innym pomóżdżyć 😎

//const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath('case-1.json'));
//const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath('case-2.json'));
// const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath('case-3.json'));
 const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath('case-4.json'));

console.log('WK_Matrix');
console.log(displayMatrix(WK_Matrix, -1));
console.log('WQ_Matrix');
console.log(displayMatrix(WQ_Matrix, -1));
console.log('X_Input_Matrix');
console.log(displayMatrix(X_Input_Matrix, -1));

const x1_vector = X_Input_Matrix[0];
console.log('x1_vector');
console.log(displayVector(x1_vector, -1));

// Calculate Q matrix: Q = X * WQ
const Q_Matrix = multiplyMatrices(X_Input_Matrix, WQ_Matrix);
console.log('Q_Matrix');
console.log(displayMatrix(Q_Matrix, -1));

// Calculate K matrix: K = X * WK
const K_Matrix = multiplyMatrices(X_Input_Matrix, WK_Matrix);
console.log('K_Matrix');
console.log(displayMatrix(K_Matrix, -1));

// Calculate S matrix: S = Q * K^T
const S_Matrix = multiplyMatrices(Q_Matrix, transpose(K_Matrix));
console.log('S_Matrix');
console.log(displayMatrix(S_Matrix, -1));

// Calculate scaled scores: Scaled_Scores = Scores / √d_k
// d_k is the key dimension (number of columns in K_Matrix)
const d_k = K_Matrix[0].length;
const sqrt_d_k = Math.sqrt(d_k);
console.log(`d_k (key dimension): ${d_k}`);
console.log(`√d_k: ${sqrt_d_k}`);

// Scale the scores by dividing each element by √d_k
const Scaled_S_Matrix: Matrix = S_Matrix.map(row => 
  row.map(value => value / sqrt_d_k)
);
console.log('Scaled_S_Matrix');
console.log(displayMatrix(Scaled_S_Matrix, -1));

// Apply softmax function row-wise to get attention matrix
// Softmax formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// The subtraction of max(x) is for numerical stability
function softmaxRow(row: Vector): Vector {
  const maxValue = Math.max(...row);
  const expValues = row.map(value => Math.exp(value - maxValue));
  const sumExp = expValues.reduce((sum, val) => sum + val, 0);
  return expValues.map(expVal => expVal / sumExp);
}

function softmaxMatrix(matrix: Matrix): Matrix {
  return matrix.map(row => softmaxRow(row));
}

const Attention_Matrix = softmaxMatrix(Scaled_S_Matrix);
console.log('Attention_Matrix (softmax applied to Scaled_S_Matrix)');
console.log(displayMatrix(Attention_Matrix, -1));

// przypomnienie zadania: naley policzyć "attention matrix S"
