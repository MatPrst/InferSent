# InferSent

This repo is a replication of the some of the results obtained by [Conneau et al. (2017)](https://arxiv.org/abs/1705.02364). It was part of the course Advanced Topics in Computational Semantics ([see](https://cl-illc.github.io/semantics-2021/)) at the University of Amsterdam.

## Dependencies

First, you will need to have [anaconda](https://docs.anaconda.com/anaconda/install/linux/) installed. Afterwards, you can install all the dependencies use the [environment file](env.yml).

```bash
pip install foobar
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)