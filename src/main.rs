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
    println!("{:?}", labels); // ["Obesity", "Normal"]
    assert_eq!(labels, ["Obesity", "Normal"]);

    // Convert Data to CSV
    let s = clf.to_csv(',');
    println!("{}", s);

    // Convert from CSV
    clf.from_csv(&s, ',');
    // Predict one
    let label = clf.predict_one(&[150., 80.]);
    assert_eq!(label, "Obesity");
}
