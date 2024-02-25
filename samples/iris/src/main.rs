use knn_classifier::KnnClassifier;
fn main() {
    const IRIS_CSV: &str = "iris.csv";
    // check file exists
    if !std::path::Path::new(IRIS_CSV).exists() {
        println!("Please download iris.csv");
        return;
    }
    // read text file
    let text = std::fs::read_to_string(IRIS_CSV).unwrap();
    // load from csv
    let mut clf_csv = KnnClassifier::new(7);
    clf_csv.from_csv(&text, ',', 4, true);
    // test
    let test_data = vec![
        vec![5.1, 3.5, 1.4, 0.2],
        vec![6.2, 3.4, 5.4, 2.3],
        vec![7.7, 3.8, 6.7, 2.2],
    ];
    let result = clf_csv.predict(&test_data);
    println!("{:?} => {:?}", test_data, result);
    // --- 
    // check accuracy
    let mut clf = KnnClassifier::new(7);
    // shuffle
    lazyrand::shuffle(&mut clf_csv.items);
    // split
    let (train, test) = clf_csv.items.split_at(100);
    clf.items = train.iter().map(|it| it.clone()).collect();
    // extract test_x.data
    let test_x:Vec<Vec<f64>> = test.iter().map(|it| it.data.clone()).collect();
    let test_y = clf.predict(&test_x);
    // check accuracy
    let ok = test_y.iter().zip(test.iter()).filter(|(label,it)| **label == it.label).count();
    let acc = ok as f64 / test_y.len() as f64;
    println!("Accuracy = {}/{} = {}", ok, test_y.len(), acc); // (result) Accuracy = 49/50 = 0.98
}
