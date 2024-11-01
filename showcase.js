const data = [
    { review: "Amazing movie! The plot was gripping.", sentimentScore: 0.95, sentiment: "Positive" },
    { review: "Poor acting and confusing storyline.", sentimentScore: 0.1, sentiment: "Negative" }
];


const tableBody = document.querySelector("#data-table tbody");
data.forEach(item => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${item.review}</td><td>${item.sentimentScore}</td><td>${item.sentiment}</td>`;
    tableBody.appendChild(row);
});


const ctx = document.getElementById('sentimentChart').getContext('2d');
const sentimentChart = new Chart(ctx, {
    type: 'pie',
    data: {
        labels: ['Positive', 'Negative'],
        datasets: [{
            data: [60, 40], 
            backgroundColor: ['#4CAF50', '#F44336']
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false
    }
});
