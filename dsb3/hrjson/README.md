### human-readable json
Example: The following code snippet 
```python
import hrsjon
example = {'hey': {'here': [0, 1, 2]}, 'ho': [5, 6, 7]}
hrjson.dump(example, open('test.json', 'w'), indent=4, indent_to_level=0)
hrjson.dump(example, open('test.json', 'w'), indent=4, indent_to_level=1)
hrjson.dump(example, open('test.json', 'w'), indent=4)
# the previous line is equivalent with
# import json
# json.dump(example, open('test.json', 'w'), indent=4)
```
produces the output
```
{
    "ho": [5,6,7],
    "hey": {"here": [0,1,2]}
}

{
    "ho": [
        5,
        6,
        7
    ],
    "hey": {
        "here": [0,1,2]
    }
}

{
    "ho": [
        5,
        6,
        7
    ],
    "hey": {
        "here": [
            0,
            1,
            2
        ]
    }
}
```
The differences to the `json` module in the python stdlib are shown this
[commit](https://gitlab.com/falexwolf/dsb3a/commit/e5ffc2f32e2ff9559a2f4031d9cdf13fafaf9a5a).