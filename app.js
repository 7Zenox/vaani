
// const express = require("express");
// const bodyParser = require("body-parser");
// const ejs = require("ejs"); 
// const tf = require('@tensorflow/tfjs-node');
// //ejs requirements
// // const path = require("path");
// const { PythonShell } = require("python-shell");
// const app = express();

// app.use(express.static("public"));
// app.use(bodyParser.urlencoded({ extended: true }));

// app.set("view engine", "ejs");
// //ejs requirements
// // app.use(express.urlencoded({ extended: true }));
// // app.use(express.json());

// app.get("/", function(req,res){
//     res.render("home")
// })

// app.get("/about", function(req,res){
//     res.render("about")
// })

// app.get("/contact", function(req,res){
//     res.render( "contact")
// })

// app.get("/help", function(req,res){
//     res.render("help")
// })

// app.get("/response", function(req,res){
//     res.render("home")
// })

// // app.post("/", function(req,res){
// //   // console.log(req.body.input1)
// //   const post = {
// //     givenInput: req.body.input1,
// //   };
// // });

// app.post("/", function(req,res){
//   const query = req.body.input1;
//     console.log(query)
// })





// // app.post("/predict", (req, res) => {
// //   let query = req.body.query;

// //   let options = {
// //     mode: "text",
// //     pythonPath: "/usr/bin/python3",
// //     pythonOptions: ["-u"],
// //     scriptPath: path.join(__dirname, "/python"),
// //     args: [query],
// //   };

// //   PythonShell.run("Vaani_backend.py", options, (err, data) => {
// //     if (err) {
// //       console.log(err);
// //       res.send("Error");
// //     } else {
// //       let output = JSON.parse(data[0]);
// //       res.send(output);
// //     }
// //   });
// // });

// app.listen(3000, () => {
//   console.log("Server running on port 3000");
// });


