//! This is a library for solving classification problems using the k-NN algorithm.
//! Due to the simplicity of the algorithm, it is lightweight and well-suited for easily solving classification problems.
//!
//! # Simple Example
//!
//! The following sample is a program that determines if a person is of normal weight or Obesity, based on their height(cm) and weight(kg).
//!
//! ```rs
//! use knn_classifier::KnnClassifier;
//! fn main() {
//!     // Create the classifier
//!     let mut clf = KnnClassifier::new(3);
//!     // Learn from data
//!     clf.fit(
//!         &[&[170., 60.], &[166., 58.], &[152., 99.], &[163., 95.], &[150., 90.]],
//!         &["Normal", "Normal", "Obesity", "Obesity", "Obesity"]);
//!     // Predict
//!     let labels = clf.predict(&[&[159., 85.], &[165., 55.]]);
//!     println!("{:?}", labels); // ["Obesity", "Normal"]
//!     assert_eq!(labels, ["Obesity", "Normal"]);
//! }
//! ```
//!
//! ## Support CSV format
//!
//! The classifier can be converted to and from CSV format.
//!
//! ```rs
// Convert Data to CSV
//! let s = clf.to_csv(',');
//! println!("{}", s);
//! 
//! // Convert from CSV (Label columns is 0)
//! clf.from_csv(&s, ',', 0);
//! 
//! // Predict one
//! let label = clf.predict_one(&[150., 80.]);
//! assert_eq!(label, "Obesity");
//! ```
//!
//! # Reference
//! - [k-NN algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
//! - [k-NN algorithm (ja)](https://ja.wikipedia.org/wiki/K%E8%BF%91%E5%82%8D%E6%B3%95)
//!

// Define data type for k-nearest neighbor (k-nn) algorithm
#[derive(Debug, Clone)]
pub struct KnnItem {
    pub label: String,
    pub data: Vec<f64>,
}
// Define the classifier for k-nn
#[derive(Debug, Clone)]
pub struct KnnClassifier {
    pub k: usize,
    pub items: Vec<KnnItem>,
}
impl KnnClassifier {
    /// new classifier with k (0 or odd number)
    pub fn new(k: usize) -> KnnClassifier {
        // check k, should be odd number
        let k = if k > 0 { k } else { 5 };
        let k = if k % 2 == 1 { k } else { k + 1 };
        KnnClassifier { k, items: vec![] }
    }
    /// Function to learn from data
    pub fn fit(&mut self, data: &[&[f64]], labels: &[&str]) {
        // Append learning data and labels together into items
        data.iter().zip(labels.iter()).for_each(|(it, label)| {
            let item = KnnItem { label: label.to_string(), data: it.to_vec() };
            self.items.push(item);
        });
    }
    /// Function to add a single data point
    pub fn fit_one(&mut self, data: &[f64], label: &str) {
        let item = KnnItem { label: label.to_string(), data: data.to_vec() };
        self.items.push(item);
    }
    /// Function to predict based on a single data point
    pub fn predict_one(&self, item: &[f64]) -> String {
        // Calculate distances between the data to predict and the learned data
        let mut distances: Vec<(usize, f64)> = self.items.iter().enumerate().map(|(i, it)| {
            (i, calc_distance(&it.data, &item))
        }).collect();
        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        // Take k nearest neighbors and perform a majority vote
        let mut counter_map = std::collections::HashMap::new();
        for (i, _) in distances.iter().take(self.k) {
            let label = &self.items[*i].label;
            *counter_map.entry(label).or_insert(0) += 1;
        }
        // Return the most common label
        let label = counter_map.into_iter().max_by_key(|&(_, count)| count).unwrap().0;
        label.clone()
    }
    // Function to predict based on multiple data points
    pub fn predict(&self, items: &[Vec<f64>]) -> Vec<String> {
        items.iter().map(|it| self.predict_one(&it.to_vec())).collect()
    }
    /// convert to csv
    pub fn to_csv(&self, delimiter: char) -> String {
        let mut s = String::new();
        for it in &self.items {
            s.push_str(&it.label);
            s.push(delimiter);
            for d in &it.data {
                s.push_str(&d.to_string());
                s.push(delimiter);
            }
            s.pop();
            s.push('\n');
        }
        s
    }
    /// convert from csv
    pub fn from_csv(&mut self, s: &str, delimiter: char, label_col: usize, skip_header: bool) {
        // read csv line
        for (i, line) in s.lines().enumerate() {
            if skip_header && i == 0 { continue; }
            let line = line.trim();
            if line == "" { continue; }
            let mut it = KnnItem { label: "".to_string(), data: vec![] };
            let columns_iter = line.split(delimiter);
            for (i, d) in columns_iter.enumerate() {
                if i == label_col {
                    it.label = d.trim().to_string();
                } else {
                    it.data.push(d.trim().parse().unwrap());
                }
            }
            self.items.push(it);
        }
    }
}

// Function to calculate distance between two points
pub fn calc_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// test code
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn1() {
        // Obesity: 肥満 > normal: 標準 > thin: 痩せ
        let mut c = KnnClassifier::new(5);
        c.fit_one(&[150.0, 80.0], "肥満");
        c.fit_one(&[153.0, 69.0], "肥満");
        c.fit_one(&[153.0, 94.0], "肥満");
        c.fit_one(&[189.0, 96.0], "肥満");
        c.fit_one(&[159.0, 74.0], "肥満");
        c.fit_one(&[169.0, 64.0], "標準");
        c.fit_one(&[171.0, 64.0], "標準");
        c.fit_one(&[186.0, 59.0], "痩せ");
        c.fit_one(&[173.0, 84.0], "肥満");
        c.fit_one(&[156.0, 77.0], "肥満");
        c.fit_one(&[174.0, 46.0], "痩せ");
        c.fit_one(&[174.0, 54.0], "痩せ");
        c.fit_one(&[162.0, 77.0], "肥満");
        c.fit_one(&[151.0, 76.0], "肥満");
        c.fit_one(&[188.0, 55.0], "痩せ");
        c.fit_one(&[189.0, 97.0], "肥満");
        c.fit_one(&[173.0, 68.0], "標準");
        c.fit_one(&[174.0, 80.0], "肥満");
        c.fit_one(&[167.0, 56.0], "標準");
        c.fit_one(&[187.0, 95.0], "肥満");
        c.fit_one(&[175.0, 100.0], "肥満");
        c.fit_one(&[163.0, 73.0], "肥満");
        c.fit_one(&[158.0, 79.0], "肥満");
        c.fit_one(&[159.0, 45.0], "痩せ");
        c.fit_one(&[170.0, 45.0], "痩せ");
        c.fit_one(&[166.0, 81.0], "肥満");
        c.fit_one(&[155.0, 98.0], "肥満");
        c.fit_one(&[165.0, 50.0], "痩せ");
        c.fit_one(&[150.0, 83.0], "肥満");
        c.fit_one(&[168.0, 85.0], "肥満");
        // predict
        let lbl = c.predict_one(&[159.0, 85.0]);
        assert_eq!(lbl, "肥満");
        let lbl = c.predict_one(&[162.0, 58.0]);
        assert_eq!(lbl, "標準");
        let lbl = c.predict_one(&[183.0, 48.0]);
        assert_eq!(lbl, "痩せ");
    }
    #[test]
    fn test_knn2() {
        // Obesity: 肥満 > normal: 標準 > thin: 痩せ
        let mut c = KnnClassifier::new(5);
        c.fit(
            &[&[150.0, 80.0], &[153.0, 69.0], &[153.0, 94.0], &[189.0, 96.0], &[159.0, 74.0], &[169.0, 64.0], &[171.0, 64.0], &[186.0, 59.0], &[173.0, 84.0], &[156.0, 77.0], &[174.0, 46.0], &[174.0, 54.0], &[162.0, 77.0], &[151.0, 76.0], &[188.0, 55.0], &[189.0, 97.0], &[173.0, 68.0], &[174.0, 80.0], &[167.0, 56.0], &[187.0, 95.0], &[175.0, 100.0], &[163.0, 73.0], &[158.0, 79.0], &[159.0, 45.0], &[170.0, 45.0], &[166.0, 81.0], &[155.0, 98.0], &[165.0, 50.0], &[150.0, 83.0], &[168.0, 85.0]], 
            &["肥満", "肥満", "肥満", "肥満", "肥満", "標準", "標準", "痩せ", "肥満", "肥満", "痩せ", "痩せ", "肥満", "肥満", "痩せ", "肥満", "標準", "肥満", "標準", "肥満", "肥満", "肥満", "肥満", "痩せ", "痩せ", "肥満", "肥満", "痩せ", "肥満", "肥満"]);
        // predict
        let labels = c.predict(&[vec![159.0, 85.0], vec![162.0, 58.0], vec![183.0, 48.0]]);
        assert_eq!(labels, ["肥満", "標準", "痩せ"]);
    }
    #[test]
    fn test_knn3() {
        // set k = 0
        let mut c = KnnClassifier::new(0);
        c.fit(
            &[&[150.0, 80.0], &[153.0, 69.0], &[153.0, 94.0], &[189.0, 96.0], &[159.0, 74.0], &[169.0, 64.0], &[171.0, 64.0], &[186.0, 59.0], &[173.0, 84.0], &[156.0, 77.0], &[174.0, 46.0], &[174.0, 54.0], &[162.0, 77.0], &[151.0, 76.0], &[188.0, 55.0], &[189.0, 97.0], &[173.0, 68.0], &[174.0, 80.0], &[167.0, 56.0], &[187.0, 95.0], &[175.0, 100.0], &[163.0, 73.0], &[158.0, 79.0], &[159.0, 45.0], &[170.0, 45.0], &[166.0, 81.0], &[155.0, 98.0], &[165.0, 50.0], &[150.0, 83.0], &[168.0, 85.0]], 
            &["肥満", "肥満", "肥満", "肥満", "肥満", "標準", "標準", "痩せ", "肥満", "肥満", "痩せ", "痩せ", "肥満", "肥満", "痩せ", "肥満", "標準", "肥満", "標準", "肥満", "肥満", "肥満", "肥満", "痩せ", "痩せ", "肥満", "肥満", "痩せ", "肥満", "肥満"]);
        // predict
        let labels = c.predict(&[vec![159.0, 85.0], vec![162.0, 58.0], vec![183.0, 48.0]]);
        assert_eq!(labels, ["肥満", "標準", "痩せ"]);
    }
    #[test]
    fn test_to_csv() {
        //
        let mut c = KnnClassifier::new(5);
        c.fit_one(&[150.0, 80.0], "肥満");
        c.fit_one(&[153.0, 69.0], "肥満");
        c.fit_one(&[153.0, 94.0], "肥満");
        let s = c.to_csv(',');
        assert_eq!(s, "肥満,150,80\n肥満,153,69\n肥満,153,94\n");
        //
        let mut c = KnnClassifier::new(5);
        c.from_csv(&s, ',', 0, false);
        assert_eq!(&c.to_csv(','), "肥満,150,80\n肥満,153,69\n肥満,153,94\n");
        //
        let mut c = KnnClassifier::new(5);
        c.from_csv("肥満, 150, 80\n肥満 , 153, 69.0\n 肥満, 153, 94.0\n", ',', 0, false);
        assert_eq!(&c.to_csv(','), "肥満,150,80\n肥満,153,69\n肥満,153,94\n");
    }
}

