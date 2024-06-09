

    function getRandomSentence() {
        const randomIndex = Math.floor(Math.random() * sentences.length);
        return sentences[randomIndex];
    }

    document.addEventListener('DOMContentLoaded', function () {
        const randomSentence = getRandomSentence();
        document.getElementById('text').value = randomSentence.toUpperCase();

        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'violet',
            progressColor: 'purple'
        });
    });

    const form = document.getElementById('form');
    const recordButton = document.getElementById('record');
    const playButton = document.getElementById('play');
    const ttsButton = document.getElementById('tts');
    const resultDiv = document.getElementById('result');
    const audioInput = document.getElementById('audio');
    const textInput = document.getElementById('text');
    const segmentScoresBtn = document.getElementById('segmentScoresBtn');
    const recommendationsBtn = document.getElementById('recommendationsBtn');
    const segmentScoresModal = document.getElementById('segmentScoresModal');
    const recommendationsModal = document.getElementById('recommendationsModal');
    const aboutBtn = document.getElementById('aboutBtn');
    const aboutModal = document.getElementById('aboutModal');
    const modalClose = document.querySelectorAll('.modal-close');

    let mediaRecorder;
    let audioChunks = [];
    let wavesurfer;
    let audioContext;
    let audioBuffer;

    textInput.addEventListener('input', function () {
        if (textInput.value.trim() !== '') {
            ttsButton.disabled = false;
        } else {
            ttsButton.disabled = true;
        }
    });

    recordButton.addEventListener('click', function () {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            startRecording();
        }
    });

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function (stream) {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                recordButton.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
                playButton.disabled = true;

                audioChunks = [];

                mediaRecorder.addEventListener('dataavailable', function (event) {
                    audioChunks.push(event.data);
                });

                mediaRecorder.addEventListener('stop', function () {
                    processRecording();
                });
            });
    }

    function stopRecording() {
        mediaRecorder.stop();
        recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        playButton.disabled = false;
    }

    function processRecording() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = function () {
            const base64data = reader.result;
            audioInput.value = base64data;
            wavesurfer.loadBlob(audioBlob);

            audioBlob.arrayBuffer().then(arrayBuffer => {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                audioContext.decodeAudioData(arrayBuffer, function (buffer) {
                    audioBuffer = buffer;
                });
            });

            submitForm();
        };
    }

    playButton.addEventListener('click', function () {
        wavesurfer.playPause();
    });

    ttsButton.addEventListener('click', function () {
        const text = textInput.value.trim().toLowerCase();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        window.speechSynthesis.speak(utterance);
    });

    function classifyPerformance(score, log_likelihood) {
        const categories = { "green": 5, "blue": 4, "orange": 3, "red": 2 };

        let scoreClass = score > 0.95 ? "green" :
            score > 0.85 ? "blue" :
                score > 0.80 ? "orange" :
                    score > 0.75 ? "orange" : "red";

        let logClass = log_likelihood > -0.9 ? "green" :
            log_likelihood > -2 ? "blue" :
                log_likelihood > -3 ? "orange" :
                    log_likelihood > -4.9 ? "orange" : "red";

        let finalClass = categories[scoreClass] < categories[logClass] ? scoreClass : logClass;

        return finalClass;
    }

    function submitForm() {
        const formData = new FormData(form);
        fetch('/process', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                resultDiv.innerHTML = '';
                const segmentScoresBody = document.querySelector('#segmentScores tbody');
                segmentScoresBody.innerHTML = '';

                const performanceData = [];
                const poorlyPerformedWords = getPoorlyPerformedWords();

                data.word_segments.forEach(item => {
                    const wordSpan = document.createElement('span');
                    wordSpan.className = 'word';

                    const classification = classifyPerformance(item.score, item.log_likelihood);

                    let scoreDisplay = classification === 'red' ? '' : ` (${item.score})`;

                    wordSpan.innerHTML = `<span class="left-half">${item.word}${scoreDisplay}</span><span class="right-half">Actual</span>`;
                    wordSpan.setAttribute('data-start', item.start_time);
                    wordSpan.setAttribute('data-end', item.end_time);
                    wordSpan.setAttribute('data-word', item.word.toLowerCase());
                    wordSpan.classList.add(classification);

                    resultDiv.appendChild(wordSpan);

                    wordSpan.querySelector('.left-half').addEventListener('click', function () {
                        const start = parseFloat(wordSpan.getAttribute('data-start'));
                        const end = parseFloat(wordSpan.getAttribute('data-end'));

                        if (audioBuffer) {
                            const audioSource = audioContext.createBufferSource();
                            audioSource.buffer = audioBuffer;
                            audioSource.connect(audioContext.destination);

                            const startTime = start;
                            const duration = end - start;
                            audioSource.start(0, startTime, duration);
                        }
                    });

                    wordSpan.querySelector('.right-half').addEventListener('click', function () {
                        const word = wordSpan.getAttribute('data-word');
                        const utterance = new SpeechSynthesisUtterance(word);
                        utterance.lang = 'en-US';
                        window.speechSynthesis.speak(utterance);
                    });

                    performanceData.push({
                        word: item.word,
                        score: item.score,
                        log_likelihood: item.log_likelihood,
                        classification: classification
                    });

                    updateWordPerformance(item.word.toLowerCase(), classification);
                });

                data.segments.forEach(segment => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${segment.label}</td>
                        <td>${segment.start_time}</td>
                        <td>${segment.end_time}</td>
                        <td>${segment.score}</td>
                        <td>${segment.log_likelihood}</td>
                    `;
                    segmentScoresBody.appendChild(row);
                });

                resultDiv.style.display = 'flex';

                savePerformanceData(performanceData);
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }

    function savePerformanceData(performanceData) {
        let performanceHistory = JSON.parse(localStorage.getItem('performanceHistory')) || [];
        performanceHistory.push({
            date: new Date().toISOString(),
            data: performanceData
        });
        localStorage.setItem('performanceHistory', JSON.stringify(performanceHistory));
    }

    function getPerformanceData() {
        return JSON.parse(localStorage.getItem('performanceHistory')) || [];
    }

    function getPoorlyPerformedWords() {
        return JSON.parse(localStorage.getItem('poorlyPerformedWords')) || {};
    }

    function updateWordPerformance(word, classification) {
        const poorlyPerformedWords = getPoorlyPerformedWords();

        if (classification === 'red' || classification === 'orange') {
            if (poorlyPerformedWords[word]) {
                poorlyPerformedWords[word].count++;
            } else {
                poorlyPerformedWords[word] = { count: 1 };
            }
        } else if (classification === 'blue') {
            if (poorlyPerformedWords[word] && poorlyPerformedWords[word].count > 0) {
                poorlyPerformedWords[word].count--;
                if (poorlyPerformedWords[word].count === 0) {
                    delete poorlyPerformedWords[word];
                }
            }
        } else if (classification === 'green') {
            if (poorlyPerformedWords[word]) {
                delete poorlyPerformedWords[word];
            }
        }

        localStorage.setItem('poorlyPerformedWords', JSON.stringify(poorlyPerformedWords));
    }

    function provideRecommendations() {
        const poorlyPerformedWords = getPoorlyPerformedWords();
        const recommendationsContent = document.getElementById('recommendationsContent');
        recommendationsContent.innerHTML = '';

        const categorizedWords = {
            'Very Bad': [],
            'Need to Improve': [],
            'Good': [],
            'Excellent': []
        };

        Object.keys(poorlyPerformedWords).forEach(word => {
            const count = poorlyPerformedWords[word].count;
            const classification = classifyPerformance(count / 10, count / 10); // Simplified classification for example
            const category = getCategoryByClass(classification);
            categorizedWords[category].push(`${word} (${count})`);
        });

        Object.keys(categorizedWords).forEach(category => {
            const wordList = categorizedWords[category].join('<br>');
            const categoryContainer = document.createElement('div');
            categoryContainer.classList.add('category-container');

            categoryContainer.innerHTML = `
                <div class="category-title">${category}</div>
                <div class="word-list">${wordList}</div>
                <button class="practice-button" onclick="practiceCategory('${category}')">Practice Now</button>
            `;

            recommendationsContent.appendChild(categoryContainer);
        });
    }

    function practiceCategory(category) {
        const poorlyPerformedWords = getPoorlyPerformedWords();
        const wordsToPractice = Object.keys(poorlyPerformedWords).filter(word => {
            const count = poorlyPerformedWords[word].count;
            const classification = classifyPerformance(count / 10, count / 10);
            return getCategoryByClass(classification) === category;
        }).join(' ');

        if (wordsToPractice) {
            document.getElementById('text').value = wordsToPractice.toUpperCase();
        }

        document.getElementById('recommendationsModal').style.display = 'none';
    }

    function getCategoryByClass(classification) {
        const categories = {
            'red': 'Very Bad',
            'orange': 'Need to Improve',
            'blue': 'Good',
            'green': 'Excellent'
        };
        return categories[classification];
    }

    document.getElementById('recommendationsBtn').addEventListener('click', function () {
        provideRecommendations();
        document.getElementById('recommendationsModal').style.display = 'flex';
    });

    document.getElementById('upload').addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file) {
            processUploadedFile(file);
        }
    });

    function processUploadedFile(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function () {
            const base64data = reader.result;
            audioInput.value = base64data;
            wavesurfer.loadBlob(file);

            file.arrayBuffer().then(arrayBuffer => {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                audioContext.decodeAudioData(arrayBuffer, function (buffer) {
                    audioBuffer = buffer;
                    playButton.disabled = false;
                });
            });

            submitForm();
        };
    }

    const toggleSidebarButton = document.getElementById('toggleSidebar');
    const nav = document.querySelector('nav');

    toggleSidebarButton.addEventListener('click', function () {
        nav.classList.toggle('active');
    });

    segmentScoresBtn.addEventListener('click', function () {
        segmentScoresModal.style.display = 'flex';
    });

    recommendationsBtn.addEventListener('click', function () {
        provideRecommendations();
        document.getElementById('recommendationsModal').style.display = 'flex';
    });

    aboutBtn.addEventListener('click', function () {
        aboutModal.style.display = 'flex';
    });

    modalClose.forEach(closeBtn => {
        closeBtn.addEventListener('click', function () {
            closeBtn.parentElement.parentElement.style.display = 'none';
        });
    });

    window.addEventListener('click', function (event) {
        if (event.target == segmentScoresModal) {
            segmentScoresModal.style.display = 'none';
        }
        if (event.target == recommendationsModal) {
            recommendationsModal.style.display = 'none';
        }
        if (event.target == aboutModal) {
            aboutModal.style.display = 'none';
        }
    });

    // Clear Local Storage Functions
    document.getElementById('clearStorage').addEventListener('click', function () {
        localStorage.clear();
        alert('All local storage data has been cleared.');
    });
