package KNN

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

// change the constant when changing the dataset
const featureNum, classNum = 784, 10

func loadData(fileName string) ([][]float64, []float64) {
	println("start to read data")
	var dataArr [][]float64
	var labelArr []float64
	fp, err := os.Open(fileName)
	if err != nil {
		println("Opening %s fails", fileName)
	}
	defer fp.Close()
	scanner := bufio.NewScanner(fp)
	var count int = 0
	for scanner.Scan() {
		fmt.Println(scanner.Text())
		curLine := strings.Split(scanner.Text(), ",")
		i, err := strconv.Atoi(curLine[0])
		if err != nil {
			println("Opening %s fails", fileName)
		}
		if i >= 5 {
			labelArr = append(labelArr, 1)
		} else {
			labelArr = append(labelArr, -1)
		}
		var ele []float64
		for i, num := range curLine[1:] {
			num, err := strconv.ParseFloat(num, 64)
			_ = err // n is now "used"
			ele = append(ele, num/255)
			_ = i
		}
		dataArr[count] = ele
		count++
	}
	return dataArr, labelArr
}

func NaiveBayes(Py []float64, Px_y [classNum][featureNum][2]float64, x []float64) int {

	P := make([]float64, classNum)

	for i := 0; i < classNum; i++ {
		var sum float64 = 0
		for j := 0; j < featureNum; j++ {
			sum += Px_y[i][j][int(x[j])]
		}
		P[i] = sum + Py[i]
	}
	return index(P, max(P))
}

func max(list []float64) float64 {
	var m float64
	for i, e := range list {
		if i == 0 || e < m {
			m = e
		}
	}
	return m
}

func index(list []float64, data float64) int {
	for i, e := range list {
		if e == data {
			return i
		}
	}
	println("Finding fails")
	return -1
}

func model_test(Py []float64, Px_y [classNum][featureNum][2]float64,
	testDataArr [][]float64, testLabelArr []float64) float64 {

	var errorCnt float64 = 0

	for i := 0; i < len(testDataArr); i++ {

		presict := NaiveBayes(Py, Px_y, testDataArr[i])

		if float64(presict) != testLabelArr[i] {
			errorCnt += 1
		}
	}
	return 1 - (errorCnt / float64(len(testDataArr)))
}

func getAllProbability(trainDataArr [][]float64,
	trainLabelArr []float64) ([]float64, [classNum][featureNum][2]float64) {

	Py := make([]float64, classNum)
	for i := 0; i < classNum; i++ {

		Py[i] = (searchSum(trainLabelArr, float64(i)) + 1) / (float64(len(trainLabelArr) + 10))

	}
	Py = log(Py)

	var Px_y [classNum][featureNum][2]float64

	for i := 0; i < len(trainLabelArr); i++ {

		label := int(trainLabelArr[i])
		x := trainDataArr[i]

		for j := 0; i < featureNum; j++ {
			Px_y[label][j][int(x[j])] += 1
		}
	}

	for label := 0; label < classNum; label++ {
		for j := 0; j < featureNum; j++ {
			Px_y0 := Px_y[label][j][0]
			Px_y1 := Px_y[label][j][1]
			Px_y[label][j][0] = math.Log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
			Px_y[label][j][1] = math.Log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))
		}
	}
	return Py, Px_y
}

func searchSum(trainLabelArr []float64, value float64) float64 {

	var sum float64 = 0
	for i := 0; i < len(trainLabelArr); i++ {

		if trainLabelArr[i] == value {
			sum += value
		}

	}
	return sum
}

func log(arr []float64) []float64 {
	for i := 0; i < len(arr); i++ {
		arr[i] = math.Log(arr[i])
	}
	return arr
}

func main() {

	start := time.Now()

	println("start read transSet")
	trainDataArr, trainLabelArr := loadData("Mnist/mnist_train.csv")

	println("start read testSet")
	testDataArr, testLabelArr := loadData("../Mnist/mnist_test.csv")

	println("start to train")
	Py, Px_y := getAllProbability(trainDataArr, trainLabelArr)

	println("start to test")
	accuracy := model_test(Py, Px_y, testDataArr, testLabelArr)

	println("the accuracy is: %v", accuracy)

	println("time span: %v", time.Now().Sub(start))
}
