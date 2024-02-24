# k-nn classifier

This is a library for solving classification problems using the k-nearest neighbor (k-nn) algorithm.
Due to the simplicity of the algorithm, it is lightweight and well-suited for easily solving classification problems.

## Install

```sh
cargo add knn_classifier
```

## Simple Example

The following sample is a program that determines if a person is of normal weight or fat, based on their height(cm) and weight(kg).

```rs
use knn_classifier::KnnClassifier;
fn main() {
    // Create the classifier
    let mut clf = KnnClassifier::new(3);
    // Learn from data
    clf.fit(
        &[&[170., 60.], &[166., 58.], &[152., 99.], &[163., 95.], &[150., 90.]],
        &["Normal", "Normal", "Obesity", "Obesity", "Obesity"]);
    // Predict
    let labels = clf.predict(&[&[159., 85.], &[165., 55.]]);
    println!("{:?}", labels); // ["Fat", "Normal"]
    assert_eq!(labels, ["Obesity", "Normal"]);
}
```

## Support CSV format

The classifier can be converted to and from CSV format.

```rs
// Convert Data to CSV
let s = clf.to_csv(',');
println!("{}", s);

// Convert from CSV
clf.from_csv(&s, ',');

// Predict one
let label = clf.predict_one(&[150., 80.]);
assert_eq!(label, "Obesity");
```

## Reference

- [k-NN algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [k-NN algorithm (ja)](https://ja.wikipedia.org/wiki/K%E8%BF%91%E5%82%8D%E6%B3%95)
