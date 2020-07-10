package KNN

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/floats"
)

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

func calcDist(x1, x2 []float64) float64 {

	return floats.Distance(x1, x2, 2)

}

func getClosest(trainDataMat [][]float64, trainLabelMat []float64, x []float64, topK int) int {

	distList := make([]float64, len(trainLabelMat))

	for i := 0; i < len(trainLabelMat); i++ {

		x1 := trainDataMat[i]
		curDist := calcDist(x1, x)
		distList[i] = curDist
	}
	sort.Float64s(distList)

	topKList := distList[:topK]

	labelList := make([]float64, 10)

	for i, index := range topKList {
		_ = i
		labelList[int(trainLabelMat[int(index)])] += 1
	}
	return index(labelList, max(labelList))
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

func model_test(trainDataArr [][]float64, trainLabelArr []float64,
	testDataArr [][]float64, testLabelArr []float64, topK int) float64 {

	var errorCnt float64 = 0
	trainDataMat := trainDataArr
	trainLabelMat := trainLabelArr
	testDataMat := testDataArr
	testLabelMat := testLabelArr
	for i := 0; i < 200; i++ {

		println("test %d:%d", i, 200)

		x := testDataMat[i]

		y := getClosest(trainDataMat, trainLabelMat, x, topK)

		if y != int(testLabelMat[i]) {
			errorCnt += 1
		}
	}
	return 1 - (errorCnt / float64(200))
}

func main() {

	start := time.Now()

	trainDataArr, trainLabelArr := loadData("Mnist/mnist_train.csv")

	testDataArr, testLabelArr := loadData("../Mnist/mnist_test.csv")

	accur := model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25)

	println("accur is:%d %", accur*100)

	println("time span: %v", time.Now().Sub(start))
}
