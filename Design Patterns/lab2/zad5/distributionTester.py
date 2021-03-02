class DistributionTester:

    def __init__(self, xs):
        self.xs = xs

    def calc_percentile(self,p,strategy):
        assert len(self.xs) > 0, "you didn't generate sequence"
        return strategy(self.xs, p)



if __name__ == '__main__':
    import sequenceGenerators as gen
    dt = DistributionTester(gen.generate_fibbonaci(15))
    print(dt.xs)

    import percentileCalculator as pc
    perc = dt.calc_percentile(60,pc.nearest_rank_method)
    print("percentile : {}".format(perc))
   



