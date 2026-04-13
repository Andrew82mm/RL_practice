================================================================
  STATISTICAL TESTS
  Episodes: 1000  |  Seed: 42
================================================================

  [1/3] Сбор независимых эпизодов...
  Random: mean_lines=1.89  median=1.0  max=14                           
  Heuristic: mean_lines=8.51  median=6.0  max=65                        
  mlp_3_final: mean_lines=11.21  median=9.0  max=45                     
  cnn_gen3_transfer: mean_lines=9.05  median=8.0  max=47                

  [2/3] Попарные тесты (независимые эпизоды)...

  ────────────────────────────────────────────────────────────────
  Random  vs  Heuristic
  ────────────────────────────────────────────────────────────────

  [Lines cleared]
    Random                mean=  1.89  std= 2.29  median=  1.0
    Heuristic             mean=  8.51  std= 7.93  median=  6.0
    Mann–Whitney U:    *** (p<0.001)  (p=0.0000)
    CLES:              0.165  (P(Random>Heuristic)  | 0.5=паритет, 1.0=всегда лучше)
    Cohen's d:         -1.134  — эффект большой
    Bootstrap 95% CI:  diff=-6.62  [-7.14 .. -6.12]  ✓ значим

  [Reward]
    Random                mean= 10.96  std=12.58  median=  6.4
    Heuristic             mean= 46.02  std=41.82  median= 34.8
    Mann–Whitney U:    *** (p<0.001)  (p=0.0000)
    CLES:              0.160  (P(Random>Heuristic)  | 0.5=паритет, 1.0=всегда лучше)
    Cohen's d:         -1.135  — эффект большой
    Bootstrap 95% CI:  diff=-35.06  [-37.79 .. -32.37]  ✓ значим

  [Pieces placed]
    Random                mean= 16.49  std= 5.77  median= 14.0
    Heuristic             mean= 32.34  std=18.59  median= 26.0
    Mann–Whitney U:    *** (p<0.001)  (p=0.0000)
    CLES:              0.160  (P(Random>Heuristic)  | 0.5=паритет, 1.0=всегда лучше)
    Cohen's d:         -1.151  — эффект большой
    Bootstrap 95% CI:  diff=-15.85  [-17.07 .. -14.64]  ✓ значим

  ────────────────────────────────────────────────────────────────
  mlp_3_final  vs  Heuristic
  ────────────────────────────────────────────────────────────────

  [Lines cleared]
    mlp_3_final           mean= 11.21  std= 7.53  median=  9.0
    Heuristic             mean=  8.51  std= 7.93  median=  6.0
    Mann–Whitney U:    *** (p<0.001)  (p=0.0000)
    CLES:              0.634  (P(mlp_3_final>Heuristic)  | 0.5=паритет, 1.0=всегда лучше)
    Cohen's d:         +0.348  — эффект малый
    Bootstrap 95% CI:  diff=+2.70  [+2.02 .. +3.37]  ✓ значим

  [Reward]
    mlp_3_final           mean= 60.86  std=40.23  median= 50.0
    Heuristic             mean= 46.02  std=41.82  median= 34.8
    Mann–Whitney U:    *** (p<0.001)  (p=0.0000)
    CLES:              0.638  (P(mlp_3_final>Heuristic)  | 0.5=паритет, 1.0=всегда лучше)
    Cohen's d:         +0.361  — эффект малый
    Bootstrap 95% CI:  diff=+14.84  [+11.24 .. +18.40]  ✓ значим

  [Pieces placed]
    mlp_3_final           mean= 38.87  std=17.70  median= 35.0
    Heuristic             mean= 32.34  std=18.59  median= 26.0
    Mann–Whitney U:    *** (p<0.001)  (p=0.0000)
    CLES:              0.637  (P(mlp_3_final>Heuristic)  | 0.5=паритет, 1.0=всегда лучше)
    Cohen's d:         +0.359  — эффект малый
    Bootstrap 95% CI:  diff=+6.53  [+4.94 .. +8.10]  ✓ значим

  ────────────────────────────────────────────────────────────────
  cnn_gen3_transfer  vs  Heuristic
  ────────────────────────────────────────────────────────────────

  [Lines cleared]
    cnn_gen3_transfer     mean=  9.05  std= 5.80  median=  8.0
    Heuristic             mean=  8.51  std= 7.93  median=  6.0
    Mann–Whitney U:    *** (p<0.001)  (p=0.0000)
    CLES:              0.571  (P(cnn_gen3_transfer>Heuristic)  | 0.5=паритет, 1.0=всегда лучше)
    Cohen's d:         +0.078  — эффект пренебрежимый
    Bootstrap 95% CI:  diff=+0.54  [-0.07 .. +1.16]  ✗ включает 0

  [Reward]
    cnn_gen3_transfer     mean= 49.40  std=30.92  median= 44.1
    Heuristic             mean= 46.02  std=41.82  median= 34.8
    Mann–Whitney U:    *** (p<0.001)  (p=0.0000)
    CLES:              0.576  (P(cnn_gen3_transfer>Heuristic)  | 0.5=паритет, 1.0=всегда лучше)
    Cohen's d:         +0.092  — эффект пренебрежимый
    Bootstrap 95% CI:  diff=+3.38  [+0.15 .. +6.66]  ✓ значим

  [Pieces placed]
    cnn_gen3_transfer     mean= 34.09  std=13.82  median= 32.0
    Heuristic             mean= 32.34  std=18.59  median= 26.0
    Mann–Whitney U:    *** (p<0.001)  (p=0.0000)
    CLES:              0.579  (P(cnn_gen3_transfer>Heuristic)  | 0.5=паритет, 1.0=всегда лучше)
    Cohen's d:         +0.107  — эффект пренебрежимый
    Bootstrap 95% CI:  diff=+1.75  [+0.30 .. +3.22]  ✓ значим


  [3/3] Парные тесты (одинаковые начальные условия)...
                                                                        

  ────────────────────────────────────────────────────────────────
  PAIRED TEST: Random  vs  Heuristic  (одинаковые эпизоды)
  ────────────────────────────────────────────────────────────────
  Смысл: seed зафиксирован → одинаковые начальные фигуры.
  Убираем дисперсию от 'везения' — остаётся только качество агента.

  [Lines cleared]
    Wilcoxon signed-rank:  *** (p<0.001)  (p=0.0000)
    Медиана разности:      -4.00  (положительное = Random лучше)
    Win rate Random:        14.0%  эпизодов где Random > Heuristic
    Среднее A-B:           -6.21

  [Reward]
    Wilcoxon signed-rank:  *** (p<0.001)  (p=0.0000)
    Медиана разности:      -23.94  (положительное = Random лучше)
    Win rate Random:        16.3%  эпизодов где Random > Heuristic
    Среднее A-B:           -33.02

  [Pieces placed]
    Wilcoxon signed-rank:  *** (p<0.001)  (p=0.0000)
    Медиана разности:      -11.00  (положительное = Random лучше)
    Win rate Random:        14.7%  эпизодов где Random > Heuristic
    Среднее A-B:           -14.95

                                                                        

  ────────────────────────────────────────────────────────────────
  PAIRED TEST: mlp_3_final  vs  Heuristic  (одинаковые эпизоды)
  ────────────────────────────────────────────────────────────────
  Смысл: seed зафиксирован → одинаковые начальные фигуры.
  Убираем дисперсию от 'везения' — остаётся только качество агента.

  [Lines cleared]
    Wilcoxon signed-rank:  *** (p<0.001)  (p=0.0000)
    Медиана разности:      +2.00  (положительное = mlp_3_final лучше)
    Win rate mlp_3_final:        59.4%  эпизодов где mlp_3_final > Heuristic
    Среднее A-B:           +2.19

  [Reward]
    Wilcoxon signed-rank:  *** (p<0.001)  (p=0.0000)
    Медиана разности:      +12.69  (положительное = mlp_3_final лучше)
    Win rate mlp_3_final:        61.1%  эпизодов где mlp_3_final > Heuristic
    Среднее A-B:           +11.99

  [Pieces placed]
    Wilcoxon signed-rank:  *** (p<0.001)  (p=0.0000)
    Медиана разности:      +6.00  (положительное = mlp_3_final лучше)
    Win rate mlp_3_final:        59.4%  эпизодов где mlp_3_final > Heuristic
    Среднее A-B:           +5.40

                                                                        

  ────────────────────────────────────────────────────────────────
  PAIRED TEST: cnn_gen3_transfer  vs  Heuristic  (одинаковые эпизоды)
  ────────────────────────────────────────────────────────────────
  Смысл: seed зафиксирован → одинаковые начальные фигуры.
  Убираем дисперсию от 'везения' — остаётся только качество агента.

  [Lines cleared]
    Wilcoxon signed-rank:  *** (p<0.001)  (p=0.0010)
    Медиана разности:      +1.00  (положительное = cnn_gen3_transfer лучше)
    Win rate cnn_gen3_transfer:        54.7%  эпизодов где cnn_gen3_transfer > Heuristic
    Среднее A-B:           +0.55

  [Reward]
    Wilcoxon signed-rank:  *** (p<0.001)  (p=0.0002)
    Медиана разности:      +8.22  (положительное = cnn_gen3_transfer лучше)
    Win rate cnn_gen3_transfer:        57.3%  эпизодов где cnn_gen3_transfer > Heuristic
    Среднее A-B:           +3.34

  [Pieces placed]
    Wilcoxon signed-rank:  *** (p<0.001)  (p=0.0001)
    Медиана разности:      +3.00  (положительное = cnn_gen3_transfer лучше)
    Win rate cnn_gen3_transfer:        55.9%  эпизодов где cnn_gen3_transfer > Heuristic
    Среднее A-B:           +1.65



  ────────────────────────────────────────────────────────────────────────
  SURVIVAL TABLE — доля эпизодов достигших N линий
  ────────────────────────────────────────────────────────────────────────
  Агент                   ≥ 1  ≥ 5  ≥10  ≥15  ≥20  ≥25  ≥30
  --------------------------------------------------------------------
  Random                   66%   12%    2%    0%    0%    0%    0%
  Heuristic                94%   64%   33%   17%    9%    4%    3%
  mlp_3_final              99%   85%   49%   26%   14%    7%    3%
  cnn_gen3_transfer        99%   78%   40%   15%    6%    2%    1%
  ────────────────────────────────────────────────────────────────────────

  ────────────────────────────────────────────────────────────────
  Интерпретация:
    p < 0.05  → разница статистически значима
    CLES > 0.6 → умеренное доминирование; > 0.7 → сильное
    Cohen's d: 0.2=мало, 0.5=средне, 0.8=много
    Bootstrap CI не включает 0 → разница надёжная
    Win rate > 60% в paired test → агент стабильно лучше на любых фигурах
  ────────────────────────────────────────────────────────────────
