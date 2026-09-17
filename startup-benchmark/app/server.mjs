import http from "node:http";

const PORT = Number(process.env.PORT ?? 3000);

const server = http.createServer((request, response) => {
    response.writeHead(200, {
        "Content-Type": "application/json"
    });

    response.end(
        JSON.stringify({
            status: "ok"
        })
    );
});

server.listen(PORT, "127.0.0.1", () => {
    console.log("SERVER_READY");
});