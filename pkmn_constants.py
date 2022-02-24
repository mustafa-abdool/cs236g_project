NUM_PKMN_TYPES = 18
NUM_REDUCED_PKMN_TYPES = 13

PKMN_TYPES = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy','Fighting',
              'Fire','Flying','Ghost','Grass','Ground','Ice','Normal','Poison',
              'Psychic','Rock','Steel','Water']

REDUCED_PKMN_TYPES = ['Bug', 'Dragon', 'Electric', 'Fighting',
              'Fire','Ghost','Grass','Normal','Poison',
              'Psychic','Rock','Steel','Water']


assert len(PKMN_TYPES) == 18
assert len(REDUCED_PKMN_TYPES) == 13

CLASS_IDX_2_PKMN_TYPE_REDUCED = dict()
CLASS_IDX_2_PKMN_TYPE = dict()

for i in range(0, len(PKMN_TYPES)):
  CLASS_IDX_2_PKMN_TYPE[i] = PKMN_TYPES[i]

for i in range(0, len(REDUCED_PKMN_TYPES)):
  CLASS_IDX_2_PKMN_TYPE_REDUCED[i] = REDUCED_PKMN_TYPES[i]
