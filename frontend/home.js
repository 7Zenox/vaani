function openNav() {
    document.getElementById("mySidenav").style.width = "250px";
    document.getElementById("main").style.marginLeft = "250px";
    document.body.style.backgroundColor = "rgba(0,0,0,0.4)";
  }
  
  function closeNav() {
    document.getElementById("mySidenav").style.width = "0";
    document.getElementById("main").style.marginLeft= "0";
    document.body.style.backgroundColor = "white";
  }

  const submitForm = (value, displayer) => {
	let input = document.getElementById(value);
	let output1 = document.getElementById(displayer);
	let inputValue = input.value;

	output1.innerHTML = inputValue;
    input.value = '';
};

const form = document.getElementById('OUTPUT');

form.addEventListener('submit', (e) => {
	e.preventDefault();
	submitForm('input1', 'output1');
});

