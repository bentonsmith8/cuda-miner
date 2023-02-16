package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"
)

func main() {

	for true {
		mine()
	}
}

func mine() {

	identity := "smithbenton" // Change with lastfirst names. NO SPACES

	blines, berr := UrlToLines("https://maiti.info/anindya/services/mining/latestblock.php")
	if berr != nil {
	}
	for _, bline := range blines {
		println("Latest block: " + bline)
	}

	// original message to hash
	s := blines[0] // last block hash

	// puzzle difficulty
	difficulty := 7

	// string to match leading zeros
	m := ""
	for i := 0; i < difficulty; i++ {
		m += "0"
	}

	// random seed using time and generation of a nonce
	rand.Seed(time.Now().UnixNano())
	nonce := rand.Intn(100000000000)

	// compute first sha256
	h := sha256.New()
	p := string(s) + " " + fmt.Sprint(nonce)

	h.Write([]byte(p))

	failflag := false

	// compute sha256 until leading zeros are found
	for trials := 1; string(hex.EncodeToString(h.Sum(nil))[:difficulty]) != m; trials++ {

		h.Reset()

		nonce++
		p = string(s) + " " + fmt.Sprint(nonce)
		h.Write([]byte(p))

		if trials > 100000000 {
			failflag = true
			break
		}
	}

	if failflag {
		fmt.Println("Trial failed, trying again...")
	} else {
		fmt.Println("New block found: " + hex.EncodeToString(h.Sum(nil)))
		lines, err := UrlToLines("https://maiti.info/anindya/services/mining/newblock.php?identity=" + identity + "&nonce=" + fmt.Sprint(nonce) + "&newblock=" + hex.EncodeToString(h.Sum(nil)))
		if err != nil {
		}
		for _, line := range lines {
			println(line)
		}
	}
}

// Dependencies

func UrlToLines(url string) ([]string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return LinesFromReader(resp.Body)
}

func LinesFromReader(r io.Reader) ([]string, error) {
	var lines []string
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return lines, nil
}
