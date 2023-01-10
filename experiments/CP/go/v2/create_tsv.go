package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

func main() {
	wordlist := os.Args[1]

	duration, err := strconv.Atoi(os.Args[2])
	if err != nil {
		panic(err)
	}

	isi, err := strconv.Atoi(os.Args[3])
	if err != nil {
		panic(err)
	}

	f, err := os.Open(wordlist)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	fileScanner := bufio.NewScanner(f)
	fileScanner.Split(bufio.ScanLines)

	durationf := float64(duration) / 1000.0
	n := 0

	for fileScanner.Scan() {
		word := fileScanner.Text()
		onset := n * (duration + isi)
		onsetf := float64(onset) / 1000.0

		fmt.Printf("%s\t%f\t%f\n", word, onsetf, durationf)

		n++

	}

}
