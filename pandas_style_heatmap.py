# pandas-style correlation matrix

# see more at http://pandas.pydata.org/pandas-docs/stable/style.html

import pandas as pd
import seaborn as sns

redwine = pd.read_csv('data/wine_quality_red.csv', sep=';')

corr_rw = redwine.corr() # pairwise correlation of columns

cmap = sns.diverging_palette(5, 250, as_cmap=True) # generate a colormap object

def magnify():
    # design "hover to magnify" features for the heatmap
    return [dict(selector="th",
                 props=[("font-size", "8.5pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em"), 
                        ("text-align", "center")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
            ]

# generate heatmap
corr_rw.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '9.5pt'})\
    .set_caption("Hover to magify")\
    .set_precision(3)\
    .set_table_styles(magnify())
