package main

/**
 * Leveler Model
 * Ported from Python to Go
 * Stefan McCabe
 * Original model by Ken Comer
 *
 * This is one of the models from Ken's dissertation. He shared the code with me and I'm using it
 * to practice scheduler construction in Go.
 *
 * Known issues:
 * I am unable to replicate Ken's results for the Poisson activation regime. I produce a lower
 * mean and standard deviation; his model appears to occasionally produce a gradient of ~0.6
 * that skew the results; my model does not do this.
 */
import (
	"fmt"
	"github.com/GaryBoone/GoStats/stats"
	"github.com/Workiva/go-datastructures/queue"
	"github.com/gonum/matrix/mat64"
	"github.com/oleiade/lane"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

/* Choices */
var NumRuns = 6
var NumTurns = 20
var activationType = inversePoisson
var Pop Population
var NumOfAgents = 1000

/* activation types */
type ActivationOrder int

const (
	uniform ActivationOrder = iota
	random
	poisson
	inversePoisson
	naturalPoisson
)

func (act ActivationOrder) String() string {
	var s string
	if act == uniform {
		s = "uniform"
	} else if act == random {
		s = "random"
	} else if act == poisson {
		s = "poisson"
	} else if act == inversePoisson {
		s = "inverse poisson"
	} else if act == naturalPoisson {
		s = "natural poisson"
	}
	return s
}

/* "Classes" */

/*
 * The Python code declares an Agent class, with constructor that sets the
 * wealth parameter. It then initializes Pop as an array of Agents. I instead
 * make Agent a struct and Population a slice of Agents, then create the Populate()
 * function to initialize the agents and set their wealths unequally.
 */
type Agent struct {
	wealth float64
	lam    float64
}

type Population []Agent

type event struct {
	time  float64
	agent *Agent
}
type events []event

//implement sort.Interface
func (e events) Len() int {
	return len(e)
}
func (e events) Less(i, j int) bool {
	return e[i].time < e[j].time
}
func (e events) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}

/* Model Creation */

// Populate initializes the agent population.
func Populate() Population {
	Pop := make(Population, NumOfAgents)
	for i := 0; i < NumOfAgents; i++ {
		Pop[i].wealth = float64(i + 1)
	}
	return Pop
}

/* Model Methods */

// Asdw returns the mean and standard deviation of Population wealth.
func Asdw(Pop Population) (mean, std float64) {
	bals := make([]float64, 0)
	for i := 0; i < len(Pop); i++ {
		bals = append(bals, Pop[i].wealth)
	}
	return stats.StatsMean(bals), stats.StatsSampleStandardDeviation(bals)
}

// Proc conducts a pairwise reset of wealth.
func Proc(a, b *Agent) { //should be pointers here, yes?
	averg := math.Floor((a.wealth + b.wealth) / 2) // simulate integer divsion
	b.wealth = averg
	a.wealth = averg
}

// Randmact randomly selects a Population's worth in pairs and levels.
func Randmact() {
	for i := 0; i < NumOfAgents/2; i++ {
		Proc(&Pop[rand.Intn(NumOfAgents)], &Pop[rand.Intn(NumOfAgents)])
	}
}

// Unifact randomly selects a Population's worth in pairs and levels.
func Unifact() {
	var turnList *queue.Queue = queue.New(int64(len(Pop)))
	var wg sync.WaitGroup

	for _, x := range rand.Perm(len(Pop)) {
		err := turnList.Put(x)
		if err != nil {
			log.Fatal(err)
		}
	}

	size := int(turnList.Len()) / 2
	for i := 0; i < size; i++ {
		bag, err := turnList.Get(2)
		if err != nil {
			log.Fatal(err)
		}

		alpha := bag[0].(int)
		beta := bag[1].(int)
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			Proc(&Pop[a], &Pop[b])
		}(alpha, beta)
	}
	wg.Wait()
}

/* func Unifact() {
	turnList := make([]*Agent, len(Pop))
	//	copy(turnList, Pop)
	for i := 0; i < len(turnList); i++ {
		turnList[i] = &Pop[i]
	}
	for i := 0; i < NumOfAgents/2; i++ {

		x := rand.Intn(len(turnList))
		alpha := turnList[x]

		if x < len(turnList)-1 {
			turnList = append(turnList[:x], turnList[x+1:]...)
		} else {
			turnList = turnList[:x]
		}

		x = rand.Intn(len(turnList))
		beta := turnList[x]

		if x < len(turnList)-1 {
			turnList = append(turnList[:x], turnList[x+1:]...)
		} else {
			turnList = turnList[:x]
		}
		Proc(alpha, beta)

		if len(turnList) < 2 {
			break
		}

	}
}*/

// Poisact activates a Pop's worth in pairs chosen based on Poisson activation probabilities.
func Poisact() {
	// make activation rate inversely proportional to distance from mean
	mnw, _ := Asdw(Pop) //mean wealth, sd of wealth
	totd := 0.0         // total distance from mean
	var denom float64

	// first calculate total distance from mean of all agents
	for i := 0; i < len(Pop); i++ {
		dist := math.Abs(Pop[i].wealth - mnw)
		totd += dist
	}

	// then set lambdas based on distance
	for i := 0; i < len(Pop); i++ {
		if activationType == inversePoisson { //rich activate faster
			denom = math.Abs(Pop[i].wealth - mnw)
			if denom == 0 {
				denom = 0.0001
			}
			Pop[i].lam = totd / denom
		} else if activationType == naturalPoisson { // poor activate faster
			denom = Pop[i].wealth
			if denom == 0 {
				denom = 0.0001
			}
			Pop[i].lam = 1 / denom
		} else {
			//lambda is proportional to dist from mean;
			// those closer are activated slower
			Pop[i].lam = math.Abs(Pop[i].wealth-mnw) / totd
			//fmt.Println(Pop[i].lam)
		}
	}

	// make average lambda = 1
	Normalize()

	// KC: Based on lambda rates, create a list of activations for this turn,
	// an array that will contain time, agent tuples. I will eventually sort this on times

	aTimes := make(events, 0) // trying an array of structs instead of an array of tuples

	for i := 0; i < len(Pop); i++ {
		// find the agent's first activation time
		nextT := -1 * math.Log(rand.Float64()) / Pop[i].lam
		for nextT < 1.0 {
			// will only put the even on the scheduler if it's less than 1
			aTimes = append(aTimes, event{time: nextT, agent: &Pop[i]})
			nextT += -1 * math.Log(rand.Float64()) / Pop[i].lam
		}
	}

	sort.Sort(aTimes)
	if len(aTimes)%2 > 0 { // make sure list is even
		aTimes = aTimes[:len(aTimes)-1] // Pop
	}
	if len(aTimes) > len(Pop) {
		// truncate list to Population size
		aTimes = aTimes[:len(Pop)] // -1?
	}

	arr0 := lane.NewDeque()
	for i := 0; i < len(aTimes); i++ {
		arr0.Append(aTimes[i])
	}

	half := int(len(aTimes) / 2) // iterate pairwise
	for j := 0; j < half; j++ {
		if arr0.Size() < 2 {
			break
		}
		alpha := arr0.Shift().(event)
		beta := arr0.Shift().(event)

		Proc(alpha.agent, beta.agent)
	}
}

// Normalize sets one turn's worth of lambda rates.
func Normalize() {
	totlam := 0.0
	for i := 0; i < len(Pop); i++ { // first determine the total lambda
		totlam += Pop[i].lam
	}
	for i := 0; i < len(Pop); i++ {
		// the following increases the total activations to reasonable number
		Pop[i].lam = Pop[i].lam * float64(NumOfAgents) * 1.1 / totlam
		// reject lambda = 0
		if Pop[i].lam == 0 {
			Pop[i].lam = float64(1) / float64(NumOfAgents)
		}
	}
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	activationTypes := []ActivationOrder{uniform, random, poisson, inversePoisson, naturalPoisson}

	totalResults := make([]*mat64.Dense, 0) // approximating a 3D matrix with a slice of 2D matrices
	for _, act := range activationTypes {
		actResults := mat64.NewDense(NumRuns, NumTurns, nil) //using NumRuns instead of len(activationTypes) because I can't make a 3D Matrix
		activationType = act

		for ri := 0; ri < NumRuns; ri++ {
			//results := make([]float64, 0)
			fmt.Printf("Starting run %d with %d turns, %s activation.\n",
				ri+1, NumTurns, act)
			timenow := time.Now()
			fmt.Printf("Time is now %v, Num Agents = %d\n", timenow, NumOfAgents)

			Pop = Populate()
			_, sdw := Asdw(Pop)

			sds := make([]float64, 0)
			sds = append(sds, sdw)
			for i := 0; i < NumTurns; i++ {
				if activationType == uniform {
					Unifact()
				} else if activationType == random {
					Randmact()
				} else {
					Poisact()
					// fmt.Println("Skipping Poisson")
				}
				_, sd := Asdw(Pop)
				sds = append(sds, sd)
			}
			results := make([]float64, 0)
			results = append(results, sds...)
			actResults.SetRow(ri, results)
		}

		totalResults = append(totalResults, actResults)

	}
	fmt.Printf("\t\t\tGradient Analysis for %v runs\n", NumRuns)
	fmt.Printf("\t\t\t   Mean\t\t\t    SD\n")
	for i := 0; i < len(totalResults); i++ {
		gradients := make([]float64, 0)
		for j := 0; j < NumRuns; j++ {

			_, row := totalResults[i].Caps()
			runArray := make([]float64, row) // why is this 5?
			totalResults[i].Row(runArray, j)
			//fmt.Printf("Output: %v\n", runArray)
			//fmt.Printf("Should be: %v\n", actResults.RowView(i))
			seq_along := make([]float64, len(runArray))
			for k := 0; k < len(runArray); k++ {
				if runArray[k] == 0 {
					runArray[k] = 0.00000000001
				}
				runArray[k] = math.Log(runArray[k])
				seq_along[k] = float64(k) // +1?
			}
			var r stats.Regression
			r.UpdateArray(seq_along, runArray)
			gradient := r.Slope()
			gradients = append(gradients, gradient)
		}

		fmt.Printf("%-15s\t\t%f\t\t%f\n", activationTypes[i], stats.StatsMean(gradients), stats.StatsSampleStandardDeviation(gradients))
	}
	/*
		fmt.Println("\nDumping results matrices:")
		for i := 0; i < len(totalResults); i++ {
			fmt.Println(activationTypes[i])
			printMatrix := mat64.Formatted(totalResults[i].T(), mat64.Prefix(""))
			fmt.Println(printMatrix)
			fmt.Println()
		}
	*/
}
