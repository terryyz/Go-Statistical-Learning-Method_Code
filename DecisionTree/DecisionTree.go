package decision_tree

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func loadData(fileName string) ([][]float64, []float64) {

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
		labelArr = append(labelArr, float64(i))

		var ele []float64
		for i, num := range curLine[1:] {
			num, err := strconv.ParseFloat(num, 64)
			_ = err // n is now "used"
			if num > 128 {
				ele = append(ele, 1.0)
			} else {
				ele = append(ele, 0.0)
			}

			_ = i
		}
		dataArr[count] = ele
		count++
	}
	return dataArr, labelArr
}
