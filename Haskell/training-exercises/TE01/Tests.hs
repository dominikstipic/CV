import Test.Hspec
import Control.Monad ( mapM_ )
import TrainingExercises
--
main :: IO ()
main = hspec $ do

  describe "TE 1.1.1" $ mapM_
    ( tester "te111" te111 )
    [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3), (7, 4), (8, 4), (9, 5), (10, 5)]

  describe "TE 1.1.2" $ mapM_
    ( tester "te112" te112 )
    [(0, "out of range"), (1,"one"), (2,"two"), (3,"three"), (4,"out of range"),
    (5,"out of range"), (6,"out of range"), (7,"out of range"), (8,"out of range"),
    (9,"out of range"), (10,"out of range")
    ]

  describe "TE 1.1.3" $ mapM_
    ( tester "te113" te113 )
    [(-4, "number is in the [-4,0] range"), (-3, "number is in the [-4,0] range"),
    (-2, "number is in the [-4,0] range"), (-1, "number is in the [-4,0] range"),
    (0, "number is in the [-4,0] range"), (1, "number is in the [1,5) range"),
    (2, "number is in the [1,5) range"), (3, "number is in the [1,5) range"),
    (4, "number is in the [1,5) range"), (26, "number is in the (25,100] range"),
    (27, "number is in the (25,100] range"), (45, "number is in the (25,100] range"),
    (65, "number is in the (25,100] range"), (83, "number is in the (25,100] range"),
    (90, "number is in the (25,100] range"), (100, "number is in the (25,100] range"),
    (-5, "out of range"), (5, "out of range"), (6, "out of range"), (24, "out of range"),
    (25, "out of range"), (-6, "out of range"), (101, "out of range"), (102, "out of range")
    ]

  describe "TE 1.2.1 String" $ mapM_
    ( tester2 "te121" te121 )
    [("abc", "de", "abc"), ("ab", "def", "def"), ("ab", "cd", "abcd"),
    ("a longer string", "a shorter string", "a shorter string")
    ]

  describe "TE 1.2.1 Number" $ mapM_
    ( tester2 "te121" te121 )
    [([1, 2, -1], [3, 4], [1, 2, -1]), ([1, 2], [3, 4, -5], [3, 4, -5]), ([1, 2], [3, 4], [1, 2, 3, 4]),
    ([1.5, 2.3, -0.21], [3.4], [1.5, 2.3, -0.21]), ([1.5, 2.1], [30.15, 4.12, -5.0], [30.15, 4.12, -5.0]),
    ([1.12, 2.15], [3.11, 4.0], [1.12, 2.15, 3.11, 4.0])
    ]

  describe "TE 1.2.2" $ mapM_
    ( tester "te122" te122 )
    [("This should be a sentence", "This should be a sentence."), ("string", "string."),
    ("already..", "already...")
    ]

  describe "TE 1.2.3" $ mapM_
    ( tester "te123" te123 )
    [("", ""), ("This", ""), ("This sentence", ""), ("This sentence contains", ""), ("This sentence contains a", ""),
    ("This sentence contains a lot", ""), ("This sentence contains a lot of", ""),
    ("This sentence contains a lot of words.", "a"),
    ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do", "sit amet, consectetur adipiscing")
    ]

  describe "TE 1.3.1" $ mapM_
    ( tester "te131" (take 5 . te131) )
    [(3, [(3, 9), (5, 25), (7, 49), (9, 81), (11, 121)]),
    (-2, [(-2, 4), (0, 0), (2, 4), (4, 16), (6, 36)])
    ]


  describe "TE 1.3.1 contains" $ do
    it "te131 contains 3 ==> (25, 625)" $ do
      (25, 625) `elem` te131 3 `shouldBe` True
    it "te131 contains 3 ==> (111, 12321)" $ do
      (111, 12321) `elem` te131 3 `shouldBe` True
    it "te131 contains 3 ==> (25, 625)" $ do
      (5673, 32182929) `elem` te131 3 `shouldBe` True
    it "te131 contains -2 ==> (25, 625)" $ do
      (40, 1600) `elem` te131 (-2) `shouldBe` True
    it "te131 contains -2 ==> (25, 625)" $ do
      (1024, 1048576) `elem` te131 (-2) `shouldBe` True


  describe "TE 1.3.2" $ mapM_
    ( tester0 "te132" te132 )
    [([(1, "The Whispering Forest"), (2, "The Whispering Bicycle"), (3, "The Whispering River"),
    (4, "The Whispering Woman"), (5, "The Whispering Necklace"), (6, "The Incredible Forest"),
    (7, "The Incredible Bicycle"), (8, "The Incredible River"), (9, "The Incredible Woman"),
    (10, "The Incredible Necklace"), (11, "The Wild Forest"), (12, "The Wild Bicycle"),
    (13, "The Wild River"), (14, "The Wild Woman"), (15, "The Wild Necklace"),
    (16, "The Deadly Forest"), (17, "The Deadly Bicycle"), (18, "The Deadly River"),
    (19, "The Deadly Woman"), (20, "The Deadly Necklace")])
    ]




  describe "TE 1.3.3" $ mapM_
    ( tester "te133" te133 )
    [([(1, 2)], [3]), ([(1, 2), (-1, -4), (1, -6), (2, 12), (-4, 31)], [3, -5, -5, 14, 27]),
    ([(0, 0), (-1, 1), (1, -1), (2, -2), (-2, 2), (3, 12)], [0, 0, 0, 0, 0, 15])
    ]

--  Extra Exercises; uncomment the following lines if you've done extra exercises:
  
  describe "TE 1.1.4" $ mapM_
    ( tester2 "te114" te114 )
    [(1, 1, 2), (0, 1, 1), (-3, 5, -14), (3, -5, -14), (-3, -5, 16), (3, 5, 16), (7, 11, 78),
    (10, 11, 56), (5, 4, 11), (4, 5, 11),(-2, 13, -12), (-4, -5, 11), (-4, 5, -9), (4, -5, -9),
    (4, 6, 13), (-4, -6, 13), (4, -6, -11), (-4, 6, -11)
    ]

  describe "TE 1.2.4" $ mapM_
    ( tester "te124" te124 )
    [("palindrome", "palindrome"), ("longlong", "longlong"), ("sharederah", "sharederah"),
    ("longgnol", "long"), ("sharederahs", "share"), ("anavolimilovana", "anavoli"),
    ("palindrome sentence ecnetnes emordnilap", "palindrome sentence")
    ]
  
  describe "TE 1.3.4" $ mapM_
    ( tester "te134" te134 )
    [(("This sentence contains some words", "This is another sentence"), ["This", "sentence"]),
    (("What about the droid attack on the wookies", "It's a system we cannot afford to lose"), []),
    (("All words are common", "All words are common"), ["All", "words", "are", "common"])
    ]



  where
    tester0 fname f (x) =
      it ( fname ++ " ==> " ++ show x ) $ f `shouldBe` x
    tester fname f (x, y) =
      it ( fname ++ " " ++ show x ++ " ==> " ++ show y ) $ f x `shouldBe` y
    tester2 fname f (x, y, z) =
      it ( fname ++ " " ++ show x ++ ", " ++ show y ++ " ==> " ++ show z ) $ f x y `shouldBe` z

