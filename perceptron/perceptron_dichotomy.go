package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func loadData(fileName string) ([][]float64, []float64) {
	println("start to read data")
	var dataArr [][]float64
	var labelArr []float64
	fp, err := os.Open(fileName)
	defer fp.Close()
	scanner := bufio.NewScanner(fp)
	var count int = 0
	for scanner.Scan() {
		fmt.Println(scanner.Text())
		curLine := strings.Split(scanner.Text(), ",")
		i, err := strconv.Atoi(curLine[0])
		if i >= 5 {
			labelArr = append(labelArr, 1)
		} else {
			labelArr = append(labelArr, -1)
		}
		var ele []float64
		for i, num := range curLine[1:] {
			num, err := strconv.ParseFloat(num, 64)
			ele = append(ele, num/255)
		}
		dataArr[count] = ele
		count++
	}
	return dataArr, labelArr
}

func perceptron(dataArr [][]float64, labelArr []float64, iter_optional ...int) ([]float64, float64) {
	var iter int = 50
	if len(iter_optional) > 0 {
		iter = iter_optional[0]
	}

	println("start to trans")
	var data []float64
	for i, d := range dataArr {
		data = append(data, d...)
	}
	dataMat := mat.NewDense(len(dataArr), len(dataArr[0]), data)
	labelMat := mat.NewVecDense(len(labelArr), labelArr).T()
	var m, n int = dataMat.Dims()

	w := make([]float64, n)

	var b float64 = 0
	var h float64 = 0.0001

	for k := 0; k < iter; k++ {

		for i := 0; i < m; i++ {
			xi := mat.Row(nil, i, dataMat)

			yi := mat.Row(nil, i, labelMat)[0]

			if -1*(floats.Dot(w, xi)+b)*yi >= 0 {

				floats.Scale(h*yi, xi)
				floats.Add(w, xi)
				b = b + h*yi
			}
		}
		fmt.Printf("Round %d:%d training", k, iter)
	}
	return w, b
}

func model_test(dataArr [][]float64, labelArr []float64, w []float64, b float64) float64 {

	println("start to test")
	var data []float64
	for i, d := range dataArr {
		data = append(data, d...)
	}
	dataMat := mat.NewDense(len(dataArr), len(dataArr[0]), data)
	labelMat := mat.NewVecDense(len(labelArr), labelArr).T()
	var m, n int = dataMat.Dims()
	var errorCnt float64 = 0

	//w_vec := mat.NewDense(1, n, w)
	for i := 0; i < m; i++ {
		xi := mat.Row(nil, i, dataMat)

		yi := mat.Row(nil, i, labelMat)[0]

		if -1*(floats.Dot(w, xi)+b)*yi >= 0 {
			errorCnt += 1
		}
	}
	var accruRate float64 = 1 - (errorCnt / float64(m))

	return accruRate

}

func main() {

	start := time.Now()

	var trainData, trainLabel = loadData("Mnist/mnist_train.csv")

	testData, testLabel := loadData("../Mnist/mnist_test.csv")

	w, b := perceptron(trainData, trainLabel, 30)

	accruRate := model_test(testData, testLabel, w, b)

	end := time.Now()

	println("accuracy rate is: %v", accruRate)

	println("time span: %v", end.Sub(start))
}
