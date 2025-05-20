data {
}
parameters {
  real y;
}
model {
  y ~ student_t(3, 100, 3);
}
