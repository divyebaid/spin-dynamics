# General Notes

## Modification done:

- Commented properly programs

- Created a `utils.py` file to simplify all and avoid repetition (There were 2 instances of prob_over_time for example)

- Inverted line on creation and annihilation func, otherwise the Fermi sign was not calculated correctly cf; Wikipedia fock state page

![alt text](image/image.png)

- Operator number too complicated for nothing: just check if 0 or 1 at the correct location

    `idx = 2 * i if spin == 'u' else 2 * i + 1
    return int(state[idx])`

- `generate_base_states`: create diagonal matrix and with np.eye: more optimized

- `time_evol`: added (hbar=1) in the parameter so that it can be modified easily

- Renamed `prob_of_state` into `transition_probability_over_time`. Used np.vdot instead of np.dot to take into account complex numbers

## Improvements (not sure):

- `Doublons between prob_over_time` and `transition_probability_over_time`. Idk which one to chose, both might be used later, and both coded differently

## To do:

- Possibility to select the interesting curves

## Thoughts:

- Is the np.abs useful in the creation and annhilation function ?

