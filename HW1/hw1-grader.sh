#!/bin/bash
# Grading for LING28 HW1
echo Python filename:
read varname
echo 
echo \>\>\> The translations are included only to help you understand the inputs
echo \>\>\> and outputs. They should NOT be included in the input and they should
echo \>\>\> NOT appear in the output.
echo 
echo 
echo ===== Q1 =====

echo ===== Name ===

echo 
echo Input: Mi nombre es Rolando \(My name is Rolando\)
echo Expected output: Hola, Rolando. ¿Cómo estás? \(Hi, Rolando. How are you?\)
echo Program output:
python $varname "Mi nombre es Rolando"
echo 
echo Input: Me llamo Rolando. \(I am called Rolando\)
echo Expected output: Hola, Rolando. ¿Cómo estás? \(Hi, Rolando. How are you?\)
echo Program output:
python $varname "Me llamo Rolando."
echo 
echo =========== Q2 ==========

echo ===== State of mind =====

echo 
echo Input: estoy muy feliz... :\) \(i\'m happy... :\)\)
echo Expected output: ¿Porqué estás feliz? \(Why are you happy?\)
echo Program output:
python $varname "estoy muy feliz... :)"
echo 
echo Input: No estoy feliz. \(I\'m not happy\)
echo Expected output: ¿Porqué no estás feliz? \(Why aren\'t you happy?\)
echo Program output:
python $varname "No estoy feliz."
echo 
echo Input: Yo estoy un poco triste. \(I\'m a bit sad\)
echo Expected output: ¿Porqué estás triste? \(Why are you sad?\)
echo Program output:
python $varname "Yo estoy un poco triste."
echo 
echo Input: Estoy un poco triste. \(I\'m a bit sad\)
echo Expected output: ¿Porqué estás triste? \(Why are you sad?\)
echo Program output:
python $varname "Estoy un poco triste."
echo 
echo ============ Q3 ===========

echo ===== Characteristics =====

echo 
echo Input: Soy desordenado. \(I\'m messy\)
echo Expected output: ¿Porqué eres desordenado? \(Why are you messy?\)
echo Program output:
python $varname "Soy desordenado."
echo 
echo Input: Yo no soy una persona ordenada. \(I\'m not an orderly person\)
echo Expected output: ¿Porqué no eres una persona ordenada? \(Why aren\'t you an orderly person?\)
echo Program output:
python $varname "Yo no soy una persona ordenada."
echo 
echo Input: Soy una estudiante de Dartmouth. \(I\'m a Dartmouth student.\)
echo Expected output: ¿Porqué eres una estudiante de Dartmouth? \(Why are you a Dartmouth student?\)
echo Program output:
python $varname "Soy una estudiante de Dartmouth."
echo 
echo Input: Es que yo soy bastante desordenado. \(It\'s just that I\'m very messy\)
echo Expected output: ¿Porqué eres bastante desordenado? \(Why are you very messy?\)
echo Program output:
python $varname "Es que yo soy bastante desordenado."
echo 
echo ===== Q4 ========

echo ===== Family ====

echo 
echo Input: A mi mamá le gustan las orquídeas. \(My mom likes orchids\)
echo Expected output: Cuéntame más de tu mamá. \(Tell me more about your mom\)
echo Program output:
python $varname "A mi mamá le gustan las orquídeas."
echo 
echo Input: Porque mi papá prefiere los árboles. \(Because my dad prefers trees\)
echo Expected output: Cuéntame más de tu papá. \(Tell me more about your dad\)
echo Program output:
python $varname "Porque mi papá prefiere los árboles."
echo 
echo Input: Porque mi mamá quiere que yo me vaya de la casa. \(Because my mom wants me to leave the house\)
echo Expected output: Cuéntame más de tu mamá. \(Tell me more about your mom\)
echo Program output:
python $varname "Porque mi mamá quiere que yo me vaya de la casa."
echo 
echo ===== Q5 =========

echo ===== Clitics ====

echo 
echo Input: no quiero aprender a programar \(i don\'t want to learn how to program\)
echo Expected output: ¿Porqué no quieres aprender a programar? \(Why don\'t you want to learn how to program?\)
echo Program output:
python $varname "no quiero aprender a programar"
echo 
echo Input: puedo comerme diez galletas. \(i can eat like ten cookies\)
echo Expected output: ¿Porqué puedes comerte diez galletas? \(Why can you eat ten cookies?\)
echo Program output:
python $varname "puedo comerme diez galletas."
echo 
echo Input: No quiero irme de viaje! \(i don\'t want to go on a trip!\)
echo Expected output: ¿Porqué no quieres irte de viaje? \(Why don\'t you want to go on a trip?\)
echo Program output:
python $varname "No quiero irme de viaje!"
echo 
echo Input: Yo puedo caminar en la nieve. \(I can walk in the snow\)
echo Expected output: ¿Porqué puedes caminar en la nieve? \(Why can you walk in the snow?\)
echo Program output:
python $varname "Yo puedo caminar en la nieve."
echo 
echo Input: Ella dice que debo ordenar el cuarto. \(She says I need to tidy up the room\)
echo Expected output: ¿Porqué debes ordenar el cuarto? \(Why do you need to tidy up the room?\)
echo Program output:
python $varname "Ella dice que debo ordenar el cuarto."
echo 
echo Input: Bueno. No sé. Me cuesta mucho. No puedo ser ordenado. \(I mean, I dunno. It\'s hard for me to be tidy.\)
echo Expected output: ¿Porqué no puedes ser ordenado? \(Why can\'t you be tidy?\)
echo Program output:
python $varname "Bueno. No sé. Me cuesta mucho. No puedo ser ordenado."
echo 
echo ===== Q6 ====================

echo ===== Thoughts and hopes ====

echo 
echo Input: Pienso que mis amigos también hacían eso. \(I think my friends did that too\)
echo Expected output: ¿Porqué piensas eso? \(Why do you think that?\)
echo Program output:
python $varname "Pienso que mis amigos también hacían eso."
echo 
echo Input: Yo espero graduarme el año próximo. \(I hope to graduate next year\)
echo Expected output: ¿Porqué esperas eso? \(Why do you hope that?\)
echo Program output:
python $varname "Yo espero graduarme el año próximo."
echo 
echo Input: Pienso que todo empezó cuando fui a la playa por primera vez. \(I think it all began when I went to the beach for the first time\)
echo Expected output: ¿Porqué piensas eso? \(Why do you think that?\)
echo Program output:
python $varname "Pienso que todo empezó cuando fui a la playa por primera vez."
echo 
echo ===== Q7 ====================

echo ===== Asking for examples ===

echo 
echo Input: Él siempre dice lo mismo. \(He always says the same thing\)
echo Expected output: ¿Puedes darme un ejemplo específico? \(Can you give me a specific example?\)
echo Program output:
python $varname "Él siempre dice lo mismo."
echo 
echo Input: Mi hermano siempre se come las galletas. \(My brother always eats the cookies\)
echo Expected output: ¿Puedes darme un ejemplo específico? \(Can you give me a specific example?\)
echo Program output:
python $varname "Mi hermano siempre se come las galletas."
echo 
echo Input: Porque cuando era niño siempre me gustaba jugar en la naturaleza, donde todo es libre. \(Because when I was a kid I always liked playing outside, in nature, where everything is free\)
echo Expected output: ¿Puedes darme un ejemplo específico? \(Can you give me a specific example?\)
echo Program output:
python $varname "Porque cuando era niño siempre me gustaba jugar en la naturaleza, donde todo es libre."
echo 
echo ===== Q8 ====================

echo ===== Handling insults ======

echo 
echo Input: No, ¡estúpida! \(No, stupid!\)
echo Expected output: ¡Hey, sin insultos! Cálmate y cuéntame más. \(Hey, no insults! Calm down and tell me more\)
echo Program output:
python $varname "No, ¡estúpida!"
echo 
echo Input: Eres una idiota. \(You\'re an idiot\)
echo Expected output: ¡Hey, sin insultos! Cálmate y cuéntame más. \(Hey, no insults! Calm down and tell me more\)
echo Program output:
python $varname "Eres una idiota."
echo 
echo ===== Q9 ====================

echo ===== Generic reply =========

echo 
echo Input: ¡Porque me da la gana! \(Because I wanna!\)
echo Expected output: Cuéntame más \(Tell me more\)
echo Program output:
python $varname "¡Porque me da la gana!"
echo 
