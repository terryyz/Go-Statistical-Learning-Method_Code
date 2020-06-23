package KNN

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
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

func NaiveBayes(Py []float64, Px_y [][][]float64, x []float64) int {
	featrueNum := 784
	classNum := 10
	P := make([]float64, classNum)

	for i := 0; i < classNum; i++ {
		var sum float64 = 0
		for j := 0; j < featrueNum; j++ {
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

func model_test(Py []float64, Px_y [][][]float64,
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

func getAllProbability(trainDataArr [][]float64, trainLabelArr []float64) 
	([]float64, [][][]float64) {

	featrueNum := 784
	classNum := 10

	Py = make([][]float64, classNum)
	for i := 0; i < classNum; i++ {
		Py[i] = make([]float64, 1)
	}
	

}
