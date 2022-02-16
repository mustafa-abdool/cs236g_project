NUM_PKMN_TYPES = 18

PKMN_TYPES = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy','Fighting','Fire','Flying','Ghost','Grass','Ground','Ice','Normal','Poison','Psychic','Rock','Steel','Water']

assert len(PKMN_TYPES) == 18

CLASS_IDX_2_PKMN_TYPE = dict()

for i in range(0, len(PKMN_TYPES)):
  CLASS_IDX_2_PKMN_TYPE[i] = PKMN_TYPES[i]
