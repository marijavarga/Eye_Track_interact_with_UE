const express = require('express')

var app = require('express')();
const bodyParser = require('body-parser')

var http = require('http').Server(app);
var io = require('socket.io')(http);
const { response } = require('express');
const { request } = require('http');
var util = require('util');
var clients = [];

var pass = {}

//const port = 3001
//const app = express()
app.use(bodyParser.urlencoded({ extended: false }))

app.post('', (req, res) => {
  //console.log(req.body)
  // const { wrist } = req.body;
  const { username, password } = req.body;
  //const { ruka } = req.body;
  // console.log(wrist)


  if (username && password) {
    res.send('OK'); // ALL GOOD
    //console.log(objekat)
    //sendWind(password)
    setPasswor(password)
    //console.log(wrist)
  } else {
    res.status(400).send('You need to provide Username & password'); // BAD REQUEST 
  }
});

function setPasswor(password){
  pass = password;
}




//app.listen(port, () => console.log(`Example app listening on port ${port}!`))
http.listen(3000, function () {
  //console.log(request, "ovo je request")
  http.on('request', (request, response) => {
    //console.log(request)

  });
  http.on('message', (req, res) => {
    //console.log(req)
  })
  console.log('listening on *:3000');

});


io.on('connection', function (socket) {
  clients.push(socket.id);
  var clientConnectedMsg = 'User connected ' + util.inspect(socket.id) + ', total: ' + clients.length;
  console.log(clientConnectedMsg);
  socket.on('disconnect', function () {
    clients.pop(socket.id);
    var clientDisconnectedMsg = 'User disconnected ' + util.inspect(socket.id) + ', total: ' + clients.length;
    console.log(clientDisconnectedMsg);
  })
});



function getRandomInRange(min, max) {
  return Math.random() * (max - min) + min;
}

function sendWind() {
  console.log('Wind sent to user');
  const vekktor = [getRandomInRange(0, 360), getRandomInRange(0, 360), getRandomInRange(0, 360)]
  io.emit('new wind', pass);
  //
  //io.emit('new wind', str);
}
setInterval(sendWind, 500);