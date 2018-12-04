Splits obtained from `github.com/jakesnell/prototypical-networks`.

To obtain the list of alphabets from the list of characters:

```bash
cat prototypical-networks/data/omniglot/splits/vinyals/trainval.txt | \
    sed -e 's:/.*::' | sort | uniq >trainval.txt
```

We removed Gurmukhi from the `test` set.
