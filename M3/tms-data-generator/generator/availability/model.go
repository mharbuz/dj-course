package availability

// Status represents the availability slot status.
type Status string

const (
	Available Status = "AVAILABLE"
	Reserved  Status = "RESERVED"
)

// ResourceAvailability represents one time-bound availability slot for a resource.
type ResourceAvailability struct {
	ID           int
	ResourceType string // "DRIVER" or "VEHICLE"
	ResourceID   int
	ValidFrom    string // TIMESTAMP formatted for SQL
	ValidTo      string
	Status       Status
	Notes        string
}
