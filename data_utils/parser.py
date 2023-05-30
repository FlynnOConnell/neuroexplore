
# mapping the location of the food well to the food type
PLACE_NAME_MAPPING = {
    "SFN08_2018-07-23_SF.nex": {
        "FL": "cheese",
        "BL": "dchoc",
        "FR": "nuts",
        "BR": "mchoc",
    },
    "SFN08_2018-08-01_SF.nex": {
        "FL": "dchoc",
        "BL": "apple",
        "FR": "nuts",
        "BR": "mchoc",
    },
    "SFN08_2018-08-08_SF.nex": {
        "FL": "mchoc",
        "BL": "dchoc",
        "FR": "nuts",
        "BR": "apple",
    },
    "SFN08_2018-08-27_SF.nex": {
        "FL": "nuts",
        "BL": "apple",
        "FR": "mchoc",
        "BR": "brocc",
    },
    "SFN08_2018-09-04_SF.nex": {
        "FL": "brocc",
        "BL": "mchoc",
        "FR": "nuts",
        "BR": "apple",
    },
    "SFN08_2018-09-06_SF.nex": {
        "FL": "apple",
        "BL": "mchoc",
        "FR": "nuts",
        "BR": "brocc",
    },
    "SFN08_2018-09-11_SF.nex": {
        "FL": "brocc",
        "BL": "nuts",
        "FR": "mchoc",
        "BR": "apple",
    },
    "SFN08_2018-09-12_SF.nex": {
        "FL": "brocc",
        "BL": "apple",
        "FR": "mchoc",
        "BR": "nuts",
    },
    "SFN10_2018-07-23_SF.nex": {
        "FL": "nuts",
        "BL": "dchoc",
        "FR": "mchoc",
        "BR": "cheese",
    },
    "SFN10_2018-07-26_SF.nex": {
        "FL": "dchoc",
        "BL": "apple",
        "FR": "cheese",
        "BR": "nuts",
    },
    "SFN10_2018-09-05_SF.nex": {
        "FL": "mchoc",
        "BL": "brocc",
        "FR": "apple",
        "BR": "nuts",
    },
    "SFN11_2018-07-17_SF.nex": {
        "FL": "dchoc",
        "BL": "apple",
        "FR": "nuts",
        "BR": "mchoc",
        },
    "SFN11_2018-08-16_SF.nex": {
        "FL": "mchoc",
        "BL": "nuts",
        "FR": "brocc",
        "BR": "apple",
        },
    "SFN11_2018-08-23_SF.nex": {
        "FL": "apple",
        "BL": "brocc",
        "FR": "nuts",
        "BR": "mchoc",
    },
    "SFN11_2018-08-27_SF.nex": {
        "FL": "apple",
        "BL": "brocc",
        "FR": "mchoc",
        "BR": "nuts",
    },
    "SFN11_2018-08-29_SF.nex": {
        "FL": "apple",
        "BL": "mchoc",
        "FR": "brocc",
        "BR": "nuts",
    },
    "SFN11_2018-10-02_SF.nex": {
        "FL": "mchoc",
        "BL": "nuts",
        "FR": "brocc",
        "BR": "apple",
    },
    "SFN11_2018-10-03_SF.nex": {
        "FL": "mchoc",
        "BL": "brocc",
        "FR": "apple",
        "BR": "nuts",
    },
    "SFN11_2018-10-10_SF.nex": {
        "FL": "apple",
        "BL": "mchoc",
        "FR": "nuts",
        "BR": "brocc",
    },
    "SFN11_2018-10-26_SF.nex": {
        "FL": "apple",
        "BL": "mchoc",
        "FR": "brocc",
        "BR": "nuts",
    },
    "SFN11_2018-11-09_SF.nex": {
        "FL": "mchoc",
        "BL": "brocc",
        "FR": "apple",
        "BR": "nuts",
    },
    "SFN13_2018-11-30_SF.nex": {
        "FL": "mchoc",
        "BL": "empty",
        "FR": "brocc",
        "BR": "nuts",
    },
    "SFN13_2018-12-11_SF.nex": {
        "FL": "apple",
        "BL": "empty",
        "FR": "nuts",
        "BR": "mchoc",
    },
    "SFN13_2019-02-21_SF.nex": {
        "FL": "brocc",
        "BL": "nuts",
        "FR": "apple",
        "BR": "empty",
    },
    "SFN13_2019-03-29_SF.nex": {
        "FL": "apple",
        "BL": "mchoc",
        "FR": "nuts",
        "BR": "empty",
    },
    "SFN14_2018-12-07_SF.nex": {
        "FL": "nuts",
        "BL": "empty",
        "FR": "mchoc",
        "BR": "apple",
    },
    "SFN16_2019-02-05_SF.nex": {
        "FL": "apple",
        "BL": "mchoc",
        "FR": "empty",
        "BR": "brocc",
    },
    "SFN16_2019-03-25_SF.nex": {
        "FL": "nuts",
        "BL": "mchoc",
        "FR": "empty",
        "BR": "brocc",
    },
    "SFN17_2019-02-06_SF.nex": {
        "FL": "apple",
        "BL": "empty",
        "FR": "mchoc",
        "BR": "nuts",
    },
    "SFN17_2019-02-27_SF.nex": {
        "FL": "nuts",
        "BL": "empty",
        "FR": "apple",
        "BR": "mchoc",
    },
    "SFN17_2019-03-13_SF.nex": {
        "FL": "nuts",
        "BL": "empty",
        "FR": "apple",
        "BR": "mchoc",
    },
}

# mapping mislabeled food types to correct ones
INCORRECT_NAME_MAPPING = {
    'grooming': 'Grooming',
    'Grooming (1)': 'Grooming',
    'grooming (1)': 'Grooming',

    'w_apple': 'well_apple',
    'w_a': 'well_apple',
    'e_a': 'eat_apple',
    'e_apple': 'eat_apple',

    'w_nuts': 'well_nuts',
    'w_n': 'well_nuts',
    'e_nuts': 'eat_nuts',
    'e_n': 'eat_nuts',

    'w_c': 'well_mchoc',
    'w_choc': 'well_mchoc',
    'w_choco': 'well_mchoc',

    'e_c': 'eat_mchoc',
    'e_choc': 'eat_mchoc',
    'e_choco': 'eat_mchoc',

    'e_dchoc': 'eat_dchoc',
    'w_dchoc': 'well_dchoc',

    'e_brocc': 'eat_brocc',
    'w_brocc': 'well_brocc',

    'w_b': 'well_brocc',
    'w_broc': 'well_brocc',
    'e_broc': 'eat_brocc',
    'e_b': 'eat_brocc',
    'w_broccoli': 'well_brocc',
    'e_broccoli': 'eat_brocc',
    'w_e': 'well_empty',
    'w_empty': 'well_empty',
}
