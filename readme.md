# Maximal Revenue Bidding Strategy in Demand-Side Platforms
This is a repository of experiment code supporting [A Revenue-Maximizing Bidding Strategy for Demand-Side Platforms](https://ieeexplore.ieee.org/document/8725546).

For any problems, please report the issues here.

## Quirk Start
Before run the demo, please first check the GitHub project [make iPinYou data](https://github.com/wnzhang/make-ipinyou-data) for pre-processing the [iPinYou dataset](http://data.computational-advertising.org).

After pre-processing data, please update the `root` variable in [./shell/mk_train_init.sh](shell/mk_train_init.sh#L2) and [./shell/mk_train_bidder.sh](shell/mk_train_bidder.sh#L2)
by location of the processed iPinYou data.

And then please run the following code line by line
```bash
bash ./shell/mk_train_init.sh
bash ./shell/mk_train_bidder.sh
```

You can get the performance tables saved in `./result/advertiser-ID/train_log.csv` and `./result/advertiser-ID/test_log.csv` like:
```
strategy	C	V	R	omega	bids	cost	imps	clicks	roi	ctr(%)	cpc
lin	1	10000	1	[ 865951.85749593       1.        ]	614638	33425703	503714	515	0.15	0.10	64891.68
lin	1.1	10000	1	[ 107179.44253713       1.        ]	614638	6283858	139908	494	0.79	0.35	12717.79
lin	1.2	10000	1	[ 59940.66143468      1.        ]	614638	2480816	66603	484	1.95	0.73	5124.59
lin	1.3	10000	1	[ 42577.1146779      1.       ]	614638	1298534	39485	475	3.66	1.20	2733.18
lin	1.4	10000	1	[ 34483.94818101      1.        ]	614638	857311	28065	475	5.54	1.69	1804.49
lin	1.5	10000	1	[ 29625.3893217      1.       ]	614638	627807	21668	475	7.57	2.19	1321.42
lin	2	10000	1	[ 21585.01432556      1.        ]	614638	331554	12418	469	14.15	3.78	706.79
lin	5	10000	1	[ 19046.78725977      1.        ]	614638	259969	9819	469	18.04	4.78	554.19
lin	10	10000	1	[ 17986.8805105      1.       ]	614638	234151	8845	469	20.03	5.30	499.15
sqrt2	1	10000	1	[ 160535.8260962        0.00001118]	614638	45205121	614576	515	0.11	0.08	87759.89
sqrt2	1.1	10000	1	[ 5000.00011448    -0.131942  ]	614638	5737312	145053	491	0.86	0.34	11682.57
sqrt2	1.2	10000	1	[ 19026.87039911      0.39689742]	614638	2460811	66568	484	1.97	0.73	5083.27
sqrt2	1.3	10000	1	[ 5000.00952652     0.23130177]	614638	1250598	40898	475	3.80	1.16	2632.28
sqrt2	1.4	10000	1	[ 5000.0123046      0.25912969]	614638	838344	29284	475	5.67	1.62	1764.56
sqrt2	1.5	10000	1	[ 5000.00007271     0.28089247]	614638	617809	22665	475	7.69	2.10	1300.38
sqrt2	2	10000	1	[ 20351.81787503      0.68188916]	614638	338095	12677	469	13.87	3.70	720.73
sqrt2	5	10000	1	[ 5000.00006474     0.35872246]	614638	249071	9816	469	18.83	4.78	530.96
sqrt2	10	10000	1	[ 5000.     1.]	614638	84984	995	468	55.08	47.04	181.55
```

Here 'lin' and 'sqrt2' mean the 'MR1' and 'MR2' proposed in original paper, respectively.

