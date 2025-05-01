# General Notes

## Modification done:

- Commented properly programs

- Created a `utils.py` file to simplify all and avoid repetition (There were 2 instances of prob_over_time for example)

- Inverted line on creation and annihilation func, otherwise the Fermi sign was not calculated correctly cf; Wikipedia fock state page

![alt text](image/image.png)

- Operator number too complicated for nothing: just check if 0 or 1 at the correct location

    `idx = 2 * i if spin == 'u' else 2 * i + 1
    return int(state[idx])`


## Improvements (not sure):

- `generate_base_states`: create diagonal matrix and split it, probably way more efficient on larger systems.

- `prob_of_state`: np.vdot instead of np.dot but idk if any complex number...

## To do:

- Possibility to select the interesting curves

## Thoughts:

- Is the np.abs useful in the creation and annhilation function ?