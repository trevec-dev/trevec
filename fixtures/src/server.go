// Package main implements a simple HTTP server with middleware support.
package main

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

// Middleware is a function that wraps an HTTP handler.
type Middleware func(http.Handler) http.Handler

// Server holds the HTTP server configuration.
type Server struct {
	Addr        string
	middlewares []Middleware
	mux         *http.ServeMux
}

// NewServer creates a new Server with the given address.
func NewServer(addr string) *Server {
	return &Server{
		Addr: addr,
		mux:  http.NewServeMux(),
	}
}

// Use adds a middleware to the server.
func (s *Server) Use(mw Middleware) {
	s.middlewares = append(s.middlewares, mw)
}

// Handle registers a handler for the given pattern.
func (s *Server) Handle(pattern string, handler http.HandlerFunc) {
	s.mux.HandleFunc(pattern, handler)
}

// Start starts the HTTP server.
func (s *Server) Start() error {
	var handler http.Handler = s.mux
	for i := len(s.middlewares) - 1; i >= 0; i-- {
		handler = s.middlewares[i](handler)
	}

	log.Printf("Server starting on %s", s.Addr)
	return http.ListenAndServe(s.Addr, handler)
}

// LoggingMiddleware logs each request.
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

// AuthMiddleware checks for a valid session token.
func AuthMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if token == "" {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func main() {
	server := NewServer(":8080")
	server.Use(LoggingMiddleware)
	server.Handle("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "OK")
	})
	log.Fatal(server.Start())
}
