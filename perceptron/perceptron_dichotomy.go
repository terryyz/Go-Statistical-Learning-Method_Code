package main
import (
    "bufio"
    "fmt"
    "os"
    "gonum.org/v1/gonum/mat"
    "strings"
    "strconv"
)

func loadData(fileName string)([][]float64, []float64) {
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
            num, err := strconv.ParseFloat(num,64)
            ele = append(ele, num/255)
        }
        dataArr[count] = ele
        count++
    }
    return dataArr, labelArr
}

func perceptron(dataArr [][]float64, labelArr []float64, iter_optional ...int)(float64, float64) {
    var iter int = 50
    if len(iter_optional) > 0 {
        iter = iter_optional[0]
    }
    println("start to trans")
    var data []float64
    for i, d := range dataArr {
        data = append(data, d...)
    }
    dataMat := mat.NewDense(len(dataArr),len(dataArr[0]), data)
    labelMat := mat.NewVecDense(len(labelArr),labelArr).T()
    var m, n int = dataMat.Dims()
    
    w := mat.NewVecDense(n, nil)

    b := mat.NewVecDense(1, nil)

    var h_val = []float64{0.0001}
    h := mat.NewVecDense(1,h_val)

    for k := 0; k < iter; k++ {

        for i := 0; i < m; i++ {
            xi :=  mat.NewVecDense(len(dataMat[i]),dataMat[i])

            yi := mat.NewVecDense(len(labelMat[i]),labelMat[i])

            var tem mat.Dense
            if tem.Scale(-1, tem.Mul(yi, tem.Mul(tem.Mul(w, xi.T()), b))) >= 0 {
                w = w.Add(w, w.Mulh*w.Mul(yi, xi))
                b = b.Add(b, tem.Mul(h, yi))
            }
        }
        fmt.Printf("Round %d:%d training", k, iter)
    }
    return w.RawVector().Data, b.At(0, 0)
}

func model_test(dataArr [][]float64, labelArr []float64, w float64, b float64)(float64){

    println("start to test")

    dataMat := mat.NewDense(len(dataArr),len(dataArr[0]), data)
    labelMat := mat.NewVecDense(len(labelArr),labelArr).T()
    var m, n int = dataMat.Dims()
    var errorCnt float64 = 0

    w_vec := mat.NewDense(1, n, w)
    for i := 0; i < m; i++ {
        xi :=  mat.NewVecDense(len(dataMat[i]),dataMat[i])
        yi := mat.NewVecDense(len(labelMat[i]),labelMat[i])
        var result mat.Dense
        result := result.Scale(-1, result.Mul(yi, result.Add(result.Mul(w, xi.T() + b))))
        if result >= 0 { 
            errorCnt ++
        }
    }
    var accruRate float64 = 1 - (errorCnt / m)

    return accruRate

}